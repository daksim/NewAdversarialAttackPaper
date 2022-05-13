# Latest Adversarial Attack Papers
**update at 2022-05-14 06:31:24**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Anomaly Detection of Adversarial Examples using Class-conditional Generative Adversarial Networks**

cs.LG

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2105.10101v2)

**Authors**: Hang Wang, David J. Miller, George Kesidis

**Abstracts**: Deep Neural Networks (DNNs) have been shown vulnerable to Test-Time Evasion attacks (TTEs, or adversarial examples), which, by making small changes to the input, alter the DNN's decision. We propose an unsupervised attack detector on DNN classifiers based on class-conditional Generative Adversarial Networks (GANs). We model the distribution of clean data conditioned on the predicted class label by an Auxiliary Classifier GAN (AC-GAN). Given a test sample and its predicted class, three detection statistics are calculated based on the AC-GAN Generator and Discriminator. Experiments on image classification datasets under various TTE attacks show that our method outperforms previous detection methods. We also investigate the effectiveness of anomaly detection using different DNN layers (input features or internal-layer features) and demonstrate, as one might expect, that anomalies are harder to detect using features closer to the DNN's output layer.



## **2. Sample Complexity Bounds for Robustly Learning Decision Lists against Evasion Attacks**

cs.LG

To appear in the proceedings of International Joint Conference on  Artificial Intelligence (2022)

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06127v1)

**Authors**: Pascale Gourdeau, Varun Kanade, Marta Kwiatkowska, James Worrell

**Abstracts**: A fundamental problem in adversarial machine learning is to quantify how much training data is needed in the presence of evasion attacks. In this paper we address this issue within the framework of PAC learning, focusing on the class of decision lists. Given that distributional assumptions are essential in the adversarial setting, we work with probability distributions on the input data that satisfy a Lipschitz condition: nearby points have similar probability. Our key results illustrate that the adversary's budget (that is, the number of bits it can perturb on each input) is a fundamental quantity in determining the sample complexity of robust learning. Our first main result is a sample-complexity lower bound: the class of monotone conjunctions (essentially the simplest non-trivial hypothesis class on the Boolean hypercube) and any superclass has sample complexity at least exponential in the adversary's budget. Our second main result is a corresponding upper bound: for every fixed $k$ the class of $k$-decision lists has polynomial sample complexity against a $\log(n)$-bounded adversary. This sheds further light on the question of whether an efficient PAC learning algorithm can always be used as an efficient $\log(n)$-robust learning algorithm under the uniform distribution.



## **3. From IP to transport and beyond: cross-layer attacks against applications**

cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06085v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: We perform the first analysis of methodologies for launching DNS cache poisoning: manipulation at the IP layer, hijack of the inter-domain routing and probing open ports via side channels. We evaluate these methodologies against DNS resolvers in the Internet and compare them with respect to effectiveness, applicability and stealth. Our study shows that DNS cache poisoning is a practical and pervasive threat.   We then demonstrate cross-layer attacks that leverage DNS cache poisoning for attacking popular systems, ranging from security mechanisms, such as RPKI, to applications, such as VoIP. In addition to more traditional adversarial goals, most notably impersonation and Denial of Service, we show for the first time that DNS cache poisoning can even enable adversaries to bypass cryptographic defences: we demonstrate how DNS cache poisoning can facilitate BGP prefix hijacking of networks protected with RPKI even when all the other networks apply route origin validation to filter invalid BGP announcements. Our study shows that DNS plays a much more central role in the Internet security than previously assumed.   We recommend mitigations for securing the applications and for preventing cache poisoning.



## **4. Segmentation-Consistent Probabilistic Lesion Counting**

eess.IV

Accepted at Medical Imaging with Deep Learning (MIDL) 2022

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2204.05276v2)

**Authors**: Julien Schroeter, Chelsea Myers-Colet, Douglas L Arnold, Tal Arbel

**Abstracts**: Lesion counts are important indicators of disease severity, patient prognosis, and treatment efficacy, yet counting as a task in medical imaging is often overlooked in favor of segmentation. This work introduces a novel continuously differentiable function that maps lesion segmentation predictions to lesion count probability distributions in a consistent manner. The proposed end-to-end approach--which consists of voxel clustering, lesion-level voxel probability aggregation, and Poisson-binomial counting--is non-parametric and thus offers a robust and consistent way to augment lesion segmentation models with post hoc counting capabilities. Experiments on Gadolinium-enhancing lesion counting demonstrate that our method outputs accurate and well-calibrated count distributions that capture meaningful uncertainty information. They also reveal that our model is suitable for multi-task learning of lesion segmentation, is efficient in low data regimes, and is robust to adversarial attacks.



## **5. Stalloris: RPKI Downgrade Attack**

cs.CR

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.06064v1)

**Authors**: Tomas Hlavacek, Philipp Jeitner, Donika Mirdita, Haya Shulman, Michael Waidner

**Abstracts**: We demonstrate the first downgrade attacks against RPKI. The key design property in RPKI that allows our attacks is the tradeoff between connectivity and security: when networks cannot retrieve RPKI information from publication points, they make routing decisions in BGP without validating RPKI. We exploit this tradeoff to develop attacks that prevent the retrieval of the RPKI objects from the public repositories, thereby disabling RPKI validation and exposing the RPKI-protected networks to prefix hijack attacks.   We demonstrate experimentally that at least 47% of the public repositories are vulnerable against a specific version of our attacks, a rate-limiting off-path downgrade attack. We also show that all the current RPKI relying party implementations are vulnerable to attacks by a malicious publication point. This translates to 20.4% of the IPv4 address space.   We provide recommendations for preventing our downgrade attacks. However, resolving the fundamental problem is not straightforward: if the relying parties prefer security over connectivity and insist on RPKI validation when ROAs cannot be retrieved, the victim AS may become disconnected from many more networks than just the one that the adversary wishes to hijack. Our work shows that the publication points are a critical infrastructure for Internet connectivity and security. Our main recommendation is therefore that the publication points should be hosted on robust platforms guaranteeing a high degree of connectivity.



## **6. Infrared Invisible Clothing:Hiding from Infrared Detectors at Multiple Angles in Real World**

cs.CV

Accepted by CVPR 2022, ORAL

**SubmitDate**: 2022-05-12    [paper-pdf](http://arxiv.org/pdf/2205.05909v1)

**Authors**: Xiaopei Zhu, Zhanhao Hu, Siyuan Huang, Jianmin Li, Xiaolin Hu

**Abstracts**: Thermal infrared imaging is widely used in body temperature measurement, security monitoring, and so on, but its safety research attracted attention only in recent years. We proposed the infrared adversarial clothing, which could fool infrared pedestrian detectors at different angles. We simulated the process from cloth to clothing in the digital world and then designed the adversarial "QR code" pattern. The core of our method is to design a basic pattern that can be expanded periodically, and make the pattern after random cropping and deformation still have an adversarial effect, then we can process the flat cloth with an adversarial pattern into any 3D clothes. The results showed that the optimized "QR code" pattern lowered the Average Precision (AP) of YOLOv3 by 87.7%, while the random "QR code" pattern and blank pattern lowered the AP of YOLOv3 by 57.9% and 30.1%, respectively, in the digital world. We then manufactured an adversarial shirt with a new material: aerogel. Physical-world experiments showed that the adversarial "QR code" pattern clothing lowered the AP of YOLOv3 by 64.6%, while the random "QR code" pattern clothing and fully heat-insulated clothing lowered the AP of YOLOv3 by 28.3% and 22.8%, respectively. We used the model ensemble technique to improve the attack transferability to unseen models.



## **7. Using Frequency Attention to Make Adversarial Patch Powerful Against Person Detector**

cs.CV

10pages, 4 figures

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.04638v2)

**Authors**: Xiaochun Lei, Chang Lu, Zetao Jiang, Zhaoting Gong, Xiang Cai, Linjun Lu

**Abstracts**: Deep neural networks (DNNs) are vulnerable to adversarial attacks. In particular, object detectors may be attacked by applying a particular adversarial patch to the image. However, because the patch shrinks during preprocessing, most existing approaches that employ adversarial patches to attack object detectors would diminish the attack success rate on small and medium targets. This paper proposes a Frequency Module(FRAN), a frequency-domain attention module for guiding patch generation. This is the first study to introduce frequency domain attention to optimize the attack capabilities of adversarial patches. Our method increases the attack success rates of small and medium targets by 4.18% and 3.89%, respectively, over the state-of-the-art attack method for fooling the human detector while assaulting YOLOv3 without reducing the attack success rate of big targets.



## **8. The Hijackers Guide To The Galaxy: Off-Path Taking Over Internet Resources**

cs.CR

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.05473v1)

**Authors**: Tianxiang Dai, Philipp Jeitner, Haya Shulman, Michael Waidner

**Abstracts**: Internet resources form the basic fabric of the digital society. They provide the fundamental platform for digital services and assets, e.g., for critical infrastructures, financial services, government. Whoever controls that fabric effectively controls the digital society.   In this work we demonstrate that the current practices of Internet resources management, of IP addresses, domains, certificates and virtual platforms are insecure. Over long periods of time adversaries can maintain control over Internet resources which they do not own and perform stealthy manipulations, leading to devastating attacks. We show that network adversaries can take over and manipulate at least 68% of the assigned IPv4 address space as well as 31% of the top Alexa domains. We demonstrate such attacks by hijacking the accounts associated with the digital resources.   For hijacking the accounts we launch off-path DNS cache poisoning attacks, to redirect the password recovery link to the adversarial hosts. We then demonstrate that the adversaries can manipulate the resources associated with these accounts. We find all the tested providers vulnerable to our attacks.   We recommend mitigations for blocking the attacks that we present in this work. Nevertheless, the countermeasures cannot solve the fundamental problem - the management of the Internet resources should be revised to ensure that applying transactions cannot be done so easily and stealthily as is currently possible.



## **9. Sardino: Ultra-Fast Dynamic Ensemble for Secure Visual Sensing at Mobile Edge**

cs.CV

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2204.08189v2)

**Authors**: Qun Song, Zhenyu Yan, Wenjie Luo, Rui Tan

**Abstracts**: Adversarial example attack endangers the mobile edge systems such as vehicles and drones that adopt deep neural networks for visual sensing. This paper presents {\em Sardino}, an active and dynamic defense approach that renews the inference ensemble at run time to develop security against the adaptive adversary who tries to exfiltrate the ensemble and construct the corresponding effective adversarial examples. By applying consistency check and data fusion on the ensemble's predictions, Sardino can detect and thwart adversarial inputs. Compared with the training-based ensemble renewal, we use HyperNet to achieve {\em one million times} acceleration and per-frame ensemble renewal that presents the highest level of difficulty to the prerequisite exfiltration attacks. We design a run-time planner that maximizes the ensemble size in favor of security while maintaining the processing frame rate. Beyond adversarial examples, Sardino can also address the issue of out-of-distribution inputs effectively. This paper presents extensive evaluation of Sardino's performance in counteracting adversarial examples and applies it to build a real-time car-borne traffic sign recognition system. Live on-road tests show the built system's effectiveness in maintaining frame rate and detecting out-of-distribution inputs due to the false positives of a preceding YOLO-based traffic sign detector.



## **10. Developing Imperceptible Adversarial Patches to Camouflage Military Assets From Computer Vision Enabled Technologies**

cs.CV

8 pages, 4 figures, 4 tables, submitted to WCCI 2022

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2202.08892v2)

**Authors**: Chris Wise, Jo Plested

**Abstracts**: Convolutional neural networks (CNNs) have demonstrated rapid progress and a high level of success in object detection. However, recent evidence has highlighted their vulnerability to adversarial attacks. These attacks are calculated image perturbations or adversarial patches that result in object misclassification or detection suppression. Traditional camouflage methods are impractical when applied to disguise aircraft and other large mobile assets from autonomous detection in intelligence, surveillance and reconnaissance technologies and fifth generation missiles. In this paper we present a unique method that produces imperceptible patches capable of camouflaging large military assets from computer vision-enabled technologies. We developed these patches by maximising object detection loss whilst limiting the patch's colour perceptibility. This work also aims to further the understanding of adversarial examples and their effects on object detection algorithms.



## **11. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-11    [paper-pdf](http://arxiv.org/pdf/2205.01094v2)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.



## **12. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2110.15163v4)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **13. SYNFI: Pre-Silicon Fault Analysis of an Open-Source Secure Element**

cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2205.04775v1)

**Authors**: Pascal Nasahl, Miguel Osorio, Pirmin Vogel, Michael Schaffner, Timothy Trippel, Dominic Rizzo, Stefan Mangard

**Abstracts**: Fault attacks are active, physical attacks that an adversary can leverage to alter the control-flow of embedded devices to gain access to sensitive information or bypass protection mechanisms. Due to the severity of these attacks, manufacturers deploy hardware-based fault defenses into security-critical systems, such as secure elements. The development of these countermeasures is a challenging task due to the complex interplay of circuit components and because contemporary design automation tools tend to optimize inserted structures away, thereby defeating their purpose. Hence, it is critical that such countermeasures are rigorously verified post-synthesis. As classical functional verification techniques fall short of assessing the effectiveness of countermeasures, developers have to resort to methods capable of injecting faults in a simulation testbench or into a physical chip. However, developing test sequences to inject faults in simulation is an error-prone task and performing fault attacks on a chip requires specialized equipment and is incredibly time-consuming. To that end, this paper introduces SYNFI, a formal pre-silicon fault verification framework that operates on synthesized netlists. SYNFI can be used to analyze the general effect of faults on the input-output relationship in a circuit and its fault countermeasures, and thus enables hardware designers to assess and verify the effectiveness of embedded countermeasures in a systematic and semi-automatic way. To demonstrate that SYNFI is capable of handling unmodified, industry-grade netlists synthesized with commercial and open tools, we analyze OpenTitan, the first open-source secure element. In our analysis, we identified critical security weaknesses in the unprotected AES block, developed targeted countermeasures, reassessed their security, and contributed these countermeasures back to the OpenTitan repository.



## **14. Semi-Targeted Model Poisoning Attack on Federated Learning via Backward Error Analysis**

cs.LG

Published in IJCNN 2022

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.11633v2)

**Authors**: Yuwei Sun, Hideya Ochiai, Jun Sakuma

**Abstracts**: Model poisoning attacks on federated learning (FL) intrude in the entire system via compromising an edge model, resulting in malfunctioning of machine learning models. Such compromised models are tampered with to perform adversary-desired behaviors. In particular, we considered a semi-targeted situation where the source class is predetermined however the target class is not. The goal is to cause the global classifier to misclassify data of the source class. Though approaches such as label flipping have been adopted to inject poisoned parameters into FL, it has been shown that their performances are usually class-sensitive varying with different target classes applied. Typically, an attack can become less effective when shifting to a different target class. To overcome this challenge, we propose the Attacking Distance-aware Attack (ADA) to enhance a poisoning attack by finding the optimized target class in the feature space. Moreover, we studied a more challenging situation where an adversary had limited prior knowledge about a client's data. To tackle this problem, ADA deduces pair-wise distances between different classes in the latent feature space from shared model parameters based on the backward error analysis. We performed extensive empirical evaluations on ADA by varying the factor of attacking frequency in three different image classification tasks. As a result, ADA succeeded in increasing the attack performance by 1.8 times in the most challenging case with an attacking frequency of 0.01.



## **15. Fingerprinting of DNN with Black-box Design and Verification**

cs.CR

**SubmitDate**: 2022-05-10    [paper-pdf](http://arxiv.org/pdf/2203.10902v3)

**Authors**: Shuo Wang, Sharif Abuadbba, Sidharth Agarwal, Kristen Moore, Ruoxi Sun, Minhui Xue, Surya Nepal, Seyit Camtepe, Salil Kanhere

**Abstracts**: Cloud-enabled Machine Learning as a Service (MLaaS) has shown enormous promise to transform how deep learning models are developed and deployed. Nonetheless, there is a potential risk associated with the use of such services since a malicious party can modify them to achieve an adverse result. Therefore, it is imperative for model owners, service providers, and end-users to verify whether the deployed model has not been tampered with or not. Such verification requires public verifiability (i.e., fingerprinting patterns are available to all parties, including adversaries) and black-box access to the deployed model via APIs. Existing watermarking and fingerprinting approaches, however, require white-box knowledge (such as gradient) to design the fingerprinting and only support private verifiability, i.e., verification by an honest party.   In this paper, we describe a practical watermarking technique that enables black-box knowledge in fingerprint design and black-box queries during verification. The service ensures the integrity of cloud-based services through public verification (i.e. fingerprinting patterns are available to all parties, including adversaries). If an adversary manipulates a model, this will result in a shift in the decision boundary. Thus, the underlying principle of double-black watermarking is that a model's decision boundary could serve as an inherent fingerprint for watermarking. Our approach captures the decision boundary by generating a limited number of encysted sample fingerprints, which are a set of naturally transformed and augmented inputs enclosed around the model's decision boundary in order to capture the inherent fingerprints of the model. We evaluated our watermarking approach against a variety of model integrity attacks and model compression attacks.



## **16. Energy-bounded Learning for Robust Models of Code**

cs.LG

There are some flaws in our experiments, we would like to fix it and  publish a fixed version again in the very near future

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.11226v2)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.



## **17. Do You Think You Can Hold Me? The Real Challenge of Problem-Space Evasion Attacks**

cs.CR

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04293v1)

**Authors**: Harel Berger, Amit Dvir, Chen Hajaj, Rony Ronen

**Abstracts**: Android malware is a spreading disease in the virtual world. Anti-virus and detection systems continuously undergo patches and updates to defend against these threats. Most of the latest approaches in malware detection use Machine Learning (ML). Against the robustifying effort of detection systems, raise the \emph{evasion attacks}, where an adversary changes its targeted samples so that they are misclassified as benign. This paper considers two kinds of evasion attacks: feature-space and problem-space. \emph{Feature-space} attacks consider an adversary who manipulates ML features to evade the correct classification while minimizing or constraining the total manipulations. \textit{Problem-space} attacks refer to evasion attacks that change the actual sample. Specifically, this paper analyzes the gap between these two types in the Android malware domain. The gap between the two types of evasion attacks is examined via the retraining process of classifiers using each one of the evasion attack types. The experiments show that the gap between these two types of retrained classifiers is dramatic and may increase to 96\%. Retrained classifiers of feature-space evasion attacks have been found to be either less effective or completely ineffective against problem-space evasion attacks. Additionally, exploration of different problem-space evasion attacks shows that retraining of one problem-space evasion attack may be effective against other problem-space evasion attacks.



## **18. Federated Multi-Armed Bandits Under Byzantine Attacks**

cs.LG

13 pages, 15 figures

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04134v1)

**Authors**: Ilker Demirel, Yigit Yildirim, Cem Tekin

**Abstracts**: Multi-armed bandits (MAB) is a simple reinforcement learning model where the learner controls the trade-off between exploration versus exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is a recently emerging framework where a cohort of learners with heterogeneous local models play a MAB game and communicate their aggregated feedback to a parameter server to learn the global feedback model. Federated learning models are vulnerable to adversarial attacks such as model-update attacks or data poisoning. In this work, we study an FMAB problem in the presence of Byzantine clients who can send false model updates that pose a threat to the learning process. We borrow tools from robust statistics and propose a median-of-means-based estimator: Fed-MoM-UCB, to cope with the Byzantine clients. We show that if the Byzantine clients constitute at most half the cohort, it is possible to incur a cumulative regret on the order of ${\cal O} (\log T)$ with respect to an unavoidable error margin, including the communication cost between the clients and the parameter server. We analyze the interplay between the algorithm parameters, unavoidable error margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.



## **19. ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning**

cs.LG

Accepted to CVPR 2022

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04007v1)

**Authors**: Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, Chaitali Chakrabarti

**Abstracts**: This work aims to tackle Model Inversion (MI) attack on Split Federated Learning (SFL). SFL is a recent distributed training scheme where multiple clients send intermediate activations (i.e., feature map), instead of raw data, to a central server. While such a scheme helps reduce the computational load at the client end, it opens itself to reconstruction of raw data from intermediate activation by the server. Existing works on protecting SFL only consider inference and do not handle attacks during training. So we propose ResSFL, a Split Federated Learning Framework that is designed to be MI-resistant during training. It is based on deriving a resistant feature extractor via attacker-aware training, and using this extractor to initialize the client-side model prior to standard SFL training. Such a method helps in reducing the computational complexity due to use of strong inversion model in client-side adversarial training as well as vulnerability of attacks launched in early training epochs. On CIFAR-100 dataset, our proposed framework successfully mitigates MI attack on a VGG-11 model with a high reconstruction Mean-Square-Error of 0.050 compared to 0.005 obtained by the baseline system. The framework achieves 67.5% accuracy (only 1% accuracy drop) with very low computation overhead. Code is released at: https://github.com/zlijingtao/ResSFL.



## **20. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

10 pages

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.06569v2)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **21. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

cs.CR

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2205.03971v1)

**Authors**: Yan Long, Chen Yan, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Personal video conferencing has become the new norm after COVID-19 caused a seismic shift from in-person meetings and phone calls to video conferencing for daily communications and sensitive business. Video leaks participants' on-screen information because eyeglasses and other reflective objects unwittingly expose partial screen contents. Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual information gleamed from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our experimental results and models show it is possible to reconstruct and recognize on-screen text with a height as small as 10 mm with a 720p webcam. We further apply this threat model to web textual content with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution toward 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Our research proposes near-term mitigations, and justifies the importance of following the principle of least privilege for long-term defense against this attack. For privacy-sensitive scenarios, it's further recommended to develop technologies that blur all objects by default, then only unblur what is absolutely necessary to facilitate natural-looking conversations.



## **22. mFI-PSO: A Flexible and Effective Method in Adversarial Image Generation for Deep Neural Networks**

cs.LG

Accepted by 2022 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2006.03243v3)

**Authors**: Hai Shu, Ronghua Shi, Qiran Jia, Hongtu Zhu, Ziqi Chen

**Abstracts**: Deep neural networks (DNNs) have achieved great success in image classification, but can be very vulnerable to adversarial attacks with small perturbations to images. To improve adversarial image generation for DNNs, we develop a novel method, called mFI-PSO, which utilizes a Manifold-based First-order Influence measure for vulnerable image and pixel selection and the Particle Swarm Optimization for various objective functions. Our mFI-PSO can thus effectively design adversarial images with flexible, customized options on the number of perturbed pixels, the misclassification probability, and the targeted incorrect class. Experiments demonstrate the flexibility and effectiveness of our mFI-PSO in adversarial attacks and its appealing advantages over some popular methods.



## **23. IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection**

cs.CR

Accepted for publication in the 26th Pacific-Asia Conference on  Knowledge Discovery and Data Mining (PAKDD 2022)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/1809.02077v5)

**Authors**: Zilong Lin, Yong Shi, Zhi Xue

**Abstracts**: As an essential tool in security, the intrusion detection system bears the responsibility of the defense to network attacks performed by malicious traffic. Nowadays, with the help of machine learning algorithms, intrusion detection systems develop rapidly. However, the robustness of this system is questionable when it faces adversarial attacks. For the robustness of detection systems, more potential attack approaches are under research. In this paper, a framework of the generative adversarial networks, called IDSGAN, is proposed to generate the adversarial malicious traffic records aiming to attack intrusion detection systems by deceiving and evading the detection. Given that the internal structure and parameters of the detection system are unknown to attackers, the adversarial attack examples perform the black-box attacks against the detection system. IDSGAN leverages a generator to transform original malicious traffic records into adversarial malicious ones. A discriminator classifies traffic examples and dynamically learns the real-time black-box detection system. More significantly, the restricted modification mechanism is designed for the adversarial generation to preserve original attack functionalities of adversarial traffic records. The effectiveness of the model is indicated by attacking multiple algorithm-based detection models with different attack categories. The robustness is verified by changing the number of the modified features. A comparative experiment with adversarial attack baselines demonstrates the superiority of our model.



## **24. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

cs.CR

Accepted to CVPR 2022 (Oral Presentation)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2202.08602v3)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its Universal Adversarial Perturbations (UAPs). UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via contrastive learning that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence > 99.99 within only 20 fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.



## **25. Poisoning Semi-supervised Federated Learning via Unlabeled Data: Attacks and Defenses**

cs.LG

Updated Version

**SubmitDate**: 2022-05-07    [paper-pdf](http://arxiv.org/pdf/2012.04432v2)

**Authors**: Yi Liu, Xingliang Yuan, Ruihui Zhao, Cong Wang, Dusit Niyato, Yefeng Zheng

**Abstracts**: Semi-supervised Federated Learning (SSFL) has recently drawn much attention due to its practical consideration, i.e., the clients may only have unlabeled data. In practice, these SSFL systems implement semi-supervised training by assigning a "guessed" label to the unlabeled data near the labeled data to convert the unsupervised problem into a fully supervised problem. However, the inherent properties of such semi-supervised training techniques create a new attack surface. In this paper, we discover and reveal a simple yet powerful poisoning attack against SSFL. Our attack utilizes the natural characteristic of semi-supervised learning to cause the model to be poisoned by poisoning unlabeled data. Specifically, the adversary just needs to insert a small number of maliciously crafted unlabeled samples (e.g., only 0.1\% of the dataset) to infect model performance and misclassification. Extensive case studies have shown that our attacks are effective on different datasets and common semi-supervised learning methods. To mitigate the attacks, we propose a defense, i.e., a minimax optimization-based client selection strategy, to enable the server to select the clients who hold the correct label information and high-quality updates. Our defense further employs a quality-based aggregation rule to strengthen the contributions of the selected updates. Evaluations under different attack conditions show that the proposed defense can well alleviate such unlabeled poisoning attacks. Our study unveils the vulnerability of SSFL to unlabeled poisoning attacks and provides the community with potential defense methods.



## **26. Using cyber threat intelligence to support adversary understanding applied to the Russia-Ukraine conflict**

cs.CR

in Spanish language

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03469v1)

**Authors**: Oscar Sandoval Carlos

**Abstracts**: In military organizations, Cyber Threat Intelligence (CTI) supports cyberspace operations by providing the commander with essential information about the adversary, their capabilities and objectives as they operate through cyberspace. This paper, combines CTI with the MITRE ATT&CK framework in order to establish an adversary profile. In addition, it identifies the characteristics of the attack phase by analyzing the WhisperGate operation that occurred in Ukraine in January 2022, and suggests the minimum essential measures for defense.



## **27. Subverting Fair Image Search with Generative Adversarial Perturbations**

cs.LG

Accepted as a full paper at the 2022 ACM Conference on Fairness,  Accountability, and Transparency (FAccT 22)

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02414v2)

**Authors**: Avijit Ghosh, Matthew Jagielski, Christo Wilson

**Abstracts**: In this work we explore the intersection fairness and robustness in the context of ranking: when a ranking model has been calibrated to achieve some definition of fairness, is it possible for an external adversary to make the ranking model behave unfairly without having access to the model or training data? To investigate this question, we present a case study in which we develop and then attack a state-of-the-art, fairness-aware image search engine using images that have been maliciously modified using a Generative Adversarial Perturbation (GAP) model. These perturbations attempt to cause the fair re-ranking algorithm to unfairly boost the rank of images containing people from an adversary-selected subpopulation.   We present results from extensive experiments demonstrating that our attacks can successfully confer significant unfair advantage to people from the majority class relative to fairly-ranked baseline search results. We demonstrate that our attacks are robust across a number of variables, that they have close to zero impact on the relevance of search results, and that they succeed under a strict threat model. Our findings highlight the danger of deploying fair machine learning algorithms in-the-wild when (1) the data necessary to achieve fairness may be adversarially manipulated, and (2) the models themselves are not robust against attacks.



## **28. Leveraging strategic connection migration-powered traffic splitting for privacy**

cs.CR

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03326v1)

**Authors**: Mona Wang, Anunay Kulshrestha, Liang Wang, Prateek Mittal

**Abstracts**: Network-level adversaries have developed increasingly sophisticated techniques to surveil and control users' network traffic. In this paper, we exploit our observation that many encrypted protocol connections are no longer tied to device IP address (e.g., the connection migration feature in QUIC, or IP roaming in WireGuard and Mosh), due to the need for performance in a mobile-first world. We design and implement a novel framework, Connection Migration Powered Splitting (CoMPS), that utilizes these performance features for enhancing user privacy. With CoMPS, we can split traffic mid-session across network paths and heterogeneous network protocols. Such traffic splitting mitigates the ability of a network-level adversary to perform traffic analysis attacks by limiting the amount of traffic they can observe. We use CoMPS to construct a website fingerprinting defense that is resilient against traffic analysis attacks by a powerful adaptive adversary in the open-world setting. We evaluate our system using both simulated splitting data and real-world traffic that is actively split using CoMPS. In our real-world experiments, CoMPS reduces the precision and recall of VarCNN to 29.9% and 36.7% respectively in the open-world setting with 100 monitored classes. CoMPS is not only immediately deployable with any unaltered server that supports connection migration, but also incurs little overhead, decreasing throughput by only 5-20%.



## **29. Adversarial Classification under Gaussian Mechanism: Calibrating the Attack to Sensitivity**

cs.IT

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2201.09751v3)

**Authors**: Ayse Unsal, Melek Onen

**Abstracts**: This work studies anomaly detection under differential privacy (DP) with Gaussian perturbation using both statistical and information-theoretic tools. In our setting, the adversary aims to modify the content of a statistical dataset by inserting additional data without being detected by using the DP guarantee to her own benefit. To this end, we characterize information-theoretic and statistical thresholds for the first and second-order statistics of the adversary's attack, which balances the privacy budget and the impact of the attack in order to remain undetected. Additionally, we introduce a new privacy metric based on Chernoff information for classifying adversaries under differential privacy as a stronger alternative to $(\epsilon, \delta)-$ and Kullback-Leibler DP for the Gaussian mechanism. Analytical results are supported by numerical evaluations.



## **30. Learning Optimal Propagation for Graph Neural Networks**

cs.LG

7 pages, 3 figures

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02998v1)

**Authors**: Beidi Zhao, Boxin Du, Zhe Xu, Liangyue Li, Hanghang Tong

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in a variety of real-world applications by relying on the fixed graph data as input. However, the initial input graph might not be optimal in terms of specific downstream tasks, because of information scarcity, noise, adversarial attacks, or discrepancies between the distribution in graph topology, features, and groundtruth labels. In this paper, we propose a bi-level optimization-based approach for learning the optimal graph structure via directly learning the Personalized PageRank propagation matrix as well as the downstream semi-supervised node classification simultaneously. We also explore a low-rank approximation model for further reducing the time complexity. Empirical evaluations show the superior efficacy and robustness of the proposed model over all baseline methods.



## **31. Privacy-from-Birth: Protecting Sensed Data from Malicious Sensors with VERSA**

cs.CR

13 pages paper and 4 pages appendix. To be published at 2022 IEEE  Symposium on Security and Privacy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02963v1)

**Authors**: Ivan De Oliveira Nunes, Seoyeon Hwang, Sashidhar Jakkamsetti, Gene Tsudik

**Abstracts**: There are many well-known techniques to secure sensed data in IoT/CPS systems, e.g., by authenticating communication end-points, encrypting data before transmission, and obfuscating traffic patterns. Such techniques protect sensed data from external adversaries while assuming that the sensing device itself is secure. Meanwhile, both the scale and frequency of IoT-focused attacks are growing. This prompts a natural question: how to protect sensed data even if all software on the device is compromised? Ideally, in order to achieve this, sensed data must be protected from its genesis, i.e., from the time when a physical analog quantity is converted into its digital counterpart and becomes accessible to software. We refer to this property as PfB: Privacy-from-Birth.   In this work, we formalize PfB and design Verified Remote Sensing Authorization (VERSA) -- a provably secure and formally verified architecture guaranteeing that only correct execution of expected and explicitly authorized software can access and manipulate sensing interfaces, specifically, General Purpose Input/Output (GPIO), which is the usual boundary between analog and digital worlds on IoT devices. This guarantee is obtained with minimal hardware support and holds even if all device software is compromised. VERSA ensures that malware can neither gain access to sensed data on the GPIO-mapped memory nor obtain any trace thereof. VERSA is formally verified and its open-sourced implementation targets resource-constrained IoT edge devices, commonly used for sensing. Experimental results show that PfB is both achievable and affordable for such devices.



## **32. Transferring Adversarial Robustness Through Robust Representation Matching**

cs.LG

To appear at USENIX Security '22. Updated version with artifact  evaluation badges and appendix

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2202.09994v2)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.



## **33. Can collaborative learning be private, robust and scalable?**

cs.LG

Submitted to TPDP 2022

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02652v1)

**Authors**: Dmitrii Usynin, Helena Klause, Daniel Rueckert, Georgios Kaissis

**Abstracts**: We investigate the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples in train- and inference-time attacks. We explore the applications of these techniques as well as their combinations to determine which method performs best, without a significant utility trade-off. Our investigation provides a practical overview of various methods that allow one to achieve a competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation.



## **34. Holistic Approach to Measure Sample-level Adversarial Vulnerability and its Utility in Building Trustworthy Systems**

cs.CV

Accepted in CVPR Workshop 2022 on Human-centered Intelligent  Services: Safe and Trustworthy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02604v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Rohit Lal, Himanshu Patil, Anirban Chakraborty

**Abstracts**: Adversarial attack perturbs an image with an imperceptible noise, leading to incorrect model prediction. Recently, a few works showed inherent bias associated with such attack (robustness bias), where certain subgroups in a dataset (e.g. based on class, gender, etc.) are less robust than others. This bias not only persists even after adversarial training, but often results in severe performance discrepancies across these subgroups. Existing works characterize the subgroup's robustness bias by only checking individual sample's proximity to the decision boundary. In this work, we argue that this measure alone is not sufficient and validate our argument via extensive experimental analysis. It has been observed that adversarial attacks often corrupt the high-frequency components of the input image. We, therefore, propose a holistic approach for quantifying adversarial vulnerability of a sample by combining these different perspectives, i.e., degree of model's reliance on high-frequency features and the (conventional) sample-distance to the decision boundary. We demonstrate that by reliably estimating adversarial vulnerability at the sample level using the proposed holistic metric, it is possible to develop a trustworthy system where humans can be alerted about the incoming samples that are highly likely to be misclassified at test time. This is achieved with better precision when our holistic metric is used over individual measures. To further corroborate the utility of the proposed holistic approach, we perform knowledge distillation in a limited-sample setting. We observe that the student network trained with the subset of samples selected using our combined metric performs better than both the competing baselines, viz., where samples are selected randomly or based on their distances to the decision boundary.



## **35. Resilience of Bayesian Layer-Wise Explanations under Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2102.11010v3)

**Authors**: Ginevra Carbone, Guido Sanguinetti, Luca Bortolussi

**Abstracts**: We consider the problem of the stability of saliency-based explanations of Neural Network predictions under adversarial attacks in a classification task. Saliency interpretations of deterministic Neural Networks are remarkably brittle even when the attacks fail, i.e. for attacks that do not change the classification label. We empirically show that interpretations provided by Bayesian Neural Networks are considerably more stable under adversarial perturbations of the inputs and even under direct attacks to the explanations. By leveraging recent results, we also provide a theoretical explanation of this result in terms of the geometry of the data manifold. Additionally, we discuss the stability of the interpretations of high level representations of the inputs in the internal layers of a Network. Our results demonstrate that Bayesian methods, in addition to being more robust to adversarial attacks, have the potential to provide more stable and interpretable assessments of Neural Network predictions.



## **36. Robust Conversational Agents against Imperceptible Toxicity Triggers**

cs.CL

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02392v1)

**Authors**: Ninareh Mehrabi, Ahmad Beirami, Fred Morstatter, Aram Galstyan

**Abstracts**: Warning: this paper contains content that maybe offensive or upsetting. Recent research in Natural Language Processing (NLP) has advanced the development of various toxicity detection models with the intention of identifying and mitigating toxic language from existing systems. Despite the abundance of research in this area, less attention has been given to adversarial attacks that force the system to generate toxic language and the defense against them. Existing work to generate such attacks is either based on human-generated attacks which is costly and not scalable or, in case of automatic attacks, the attack vector does not conform to human-like language, which can be detected using a language model loss. In this work, we propose attacks against conversational agents that are imperceptible, i.e., they fit the conversation in terms of coherency, relevancy, and fluency, while they are effective and scalable, i.e., they can automatically trigger the system into generating toxic language. We then propose a defense mechanism against such attacks which not only mitigates the attack but also attempts to maintain the conversational flow. Through automatic and human evaluations, we show that our defense is effective at avoiding toxic language generation even against imperceptible toxicity triggers while the generated language fits the conversation in terms of coherency and relevancy. Lastly, we establish the generalizability of such a defense mechanism on language generation models beyond conversational agents.



## **37. Zero Day Threat Detection Using Graph and Flow Based Security Telemetry**

cs.CR

11 pages, 6 figures, submitting to NeurIPS 2022

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02298v1)

**Authors**: Christopher Redino, Dhruv Nandakumar, Robert Schiller, Kevin Choi, Abdul Rahman, Edward Bowen, Matthew Weeks, Aaron Shaha, Joe Nehila

**Abstracts**: Zero Day Threats (ZDT) are novel methods used by malicious actors to attack and exploit information technology (IT) networks or infrastructure. In the past few years, the number of these threats has been increasing at an alarming rate and have been costing organizations millions of dollars to remediate. The increasing expansion of network attack surfaces and the exponentially growing number of assets on these networks necessitate the need for a robust AI-based Zero Day Threat detection model that can quickly analyze petabyte-scale data for potentially malicious and novel activity. In this paper, the authors introduce a deep learning based approach to Zero Day Threat detection that can generalize, scale, and effectively identify threats in near real-time. The methodology utilizes network flow telemetry augmented with asset-level graph features, which are passed through a dual-autoencoder structure for anomaly and novelty detection respectively. The models have been trained and tested on four large scale datasets that are representative of real-world organizational networks and they produce strong results with high precision and recall values. The models provide a novel methodology to detect complex threats with low false-positive rates that allow security operators to avoid alert fatigue while drastically reducing their mean time to response with near-real-time detection. Furthermore, the authors also provide a novel, labelled, cyber attack dataset generated from adversarial activity that can be used for validation or training of other models. With this paper, the authors' overarching goal is to provide a novel architecture and training methodology for cyber anomaly detectors that can generalize to multiple IT networks with minimal to no retraining while still maintaining strong performance.



## **38. Adversarial Training for High-Stakes Reliability**

cs.LG

31 pages, 6 figures, small tweak

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.01663v2)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **39. Rethinking Classifier And Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02743v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e. not approaching the lower bound of robustness). To solve this problem, this paper first uses the Decouple Space method to divide the classifier into two parts: non-linear and linear. On this basis, this paper defines the representation vector of original example (and its space, i.e., the representation space) and uses Absolute Classification Boundaries Initialization (ACBI) iterative optimization to obtain a better attack starting point (i.e. attacking from this point can approach the lower bound of robustness faster). Particularly, this paper apply ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.



## **40. Based-CE white-box adversarial attack will not work using super-fitting**

cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02741v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep Neural Networks (DNN) are widely used in various fields due to their powerful performance, but recent studies have shown that deep learning models are vulnerable to adversarial attacks-by adding a slight perturbation to the input, the model will get wrong results. It is especially dangerous for some systems with high security requirements, so this paper proposes a new defense method by using the model super-fitting status. Model's adversarial robustness (i.e., the accuracry under adversarial attack) has been greatly improved in this status. This paper mathematically proves the effectiveness of super-fitting, and proposes a method to make the model reach this status quickly-minimaze unrelated categories scores (MUCS). Theoretically, super-fitting can resist any existing (even future) Based on CE white-box adversarial attack. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting and other nearly 50 defense models from recent conferences. The experimental results show that super-fitting method in this paper can make the trained model obtain the highest adversarial performance robustness.



## **41. Few-Shot Backdoor Attacks on Visual Object Tracking**

cs.CV

This work is accepted by the ICLR 2022. The first two authors  contributed equally to this work. In this version, we fix some typos and  errors contained in the last one. 21 pages

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2201.13178v2)

**Authors**: Yiming Li, Haoxiang Zhong, Xingjun Ma, Yong Jiang, Shu-Tao Xia

**Abstracts**: Visual object tracking (VOT) has been widely adopted in mission-critical applications, such as autonomous driving and intelligent surveillance systems. In current practice, third-party resources such as datasets, backbone networks, and training platforms are frequently used to train high-performance VOT models. Whilst these resources bring certain convenience, they also introduce new security threats into VOT models. In this paper, we reveal such a threat where an adversary can easily implant hidden backdoors into VOT models by tempering with the training process. Specifically, we propose a simple yet effective few-shot backdoor attack (FSBA) that optimizes two losses alternately: 1) a \emph{feature loss} defined in the hidden feature space, and 2) the standard \emph{tracking loss}. We show that, once the backdoor is embedded into the target model by our FSBA, it can trick the model to lose track of specific objects even when the \emph{trigger} only appears in one or a few frames. We examine our attack in both digital and physical-world settings and show that it can significantly degrade the performance of state-of-the-art VOT trackers. We also show that our attack is resistant to potential defenses, highlighting the vulnerability of VOT models to potential backdoor attacks.



## **42. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

cs.CR

This version introduces uplink overshadowing

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2106.05039v2)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Röschlin, Srdjan Čapkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver -- a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.



## **43. Can Rationalization Improve Robustness?**

cs.CL

Accepted to NAACL 2022; The code is available at  https://github.com/princeton-nlp/rationale-robustness

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2204.11790v2)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.



## **44. Don't sweat the small stuff, classify the rest: Sample Shielding to protect text classifiers against adversarial attacks**

cs.CL

9 pages, 8 figures, Accepted to NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01714v1)

**Authors**: Jonathan Rusert, Padmini Srinivasan

**Abstracts**: Deep learning (DL) is being used extensively for text classification. However, researchers have demonstrated the vulnerability of such classifiers to adversarial attacks. Attackers modify the text in a way which misleads the classifier while keeping the original meaning close to intact. State-of-the-art (SOTA) attack algorithms follow the general principle of making minimal changes to the text so as to not jeopardize semantics. Taking advantage of this we propose a novel and intuitive defense strategy called Sample Shielding. It is attacker and classifier agnostic, does not require any reconfiguration of the classifier or external resources and is simple to implement. Essentially, we sample subsets of the input text, classify them and summarize these into a final decision. We shield three popular DL text classifiers with Sample Shielding, test their resilience against four SOTA attackers across three datasets in a realistic threat setting. Even when given the advantage of knowing about our shielding strategy the adversary's attack success rate is <=10% with only one exception and often < 5%. Additionally, Sample Shielding maintains near original accuracy when applied to original texts. Crucially, we show that the `make minimal changes' approach of SOTA attackers leads to critical vulnerabilities that can be defended against with an intuitive sampling strategy.



## **45. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

cs.AI

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2112.01156v2)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.



## **46. On the uncertainty principle of neural networks**

cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01493v1)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang

**Abstracts**: Despite the successes in many fields, it is found that neural networks are vulnerability and difficult to be both accurate and robust (robust means that the prediction of the trained network stays unchanged for inputs with non-random perturbations introduced by adversarial attacks). Various empirical and analytic studies have suggested that there is more or less a trade-off between the accuracy and robustness of neural networks. If the trade-off is inherent, applications based on the neural networks are vulnerable with untrustworthy predictions. It is then essential to ask whether the trade-off is an inherent property or not. Here, we show that the accuracy-robustness trade-off is an intrinsic property whose underlying mechanism is deeply related to the uncertainty principle in quantum mechanics. We find that for a neural network to be both accurate and robust, it needs to resolve the features of the two conjugated parts $x$ (the inputs) and $\Delta$ (the derivatives of the normalized loss function $J$ with respect to $x$), respectively. Analogous to the position-momentum conjugation in quantum mechanics, we show that the inputs and their conjugates cannot be resolved by a neural network simultaneously.



## **47. Self-Ensemble Adversarial Training for Improved Robustness**

cs.LG

18 pages, 3 figures, ICLR 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2203.09678v2)

**Authors**: Hongjun Wang, Yisen Wang

**Abstracts**: Due to numerous breakthroughs in real-world applications brought by machine intelligence, deep neural networks (DNNs) are widely employed in critical applications. However, predictions of DNNs are easily manipulated with imperceptible adversarial perturbations, which impedes the further deployment of DNNs and may result in profound security and privacy implications. By incorporating adversarial samples into the training data pool, adversarial training is the strongest principled strategy against various adversarial attacks among all sorts of defense methods. Recent works mainly focus on developing new loss functions or regularizers, attempting to find the unique optimal point in the weight space. But none of them taps the potentials of classifiers obtained from standard adversarial training, especially states on the searching trajectory of training. In this work, we are dedicated to the weight states of models through the training process and devise a simple but powerful \emph{Self-Ensemble Adversarial Training} (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise. We also discuss the relationship between the ensemble of predictions from different adversarially trained models and the prediction of weight-ensembled models, as well as provide theoretical and empirical evidence that the proposed self-ensemble method provides a smoother loss landscape and better robustness than both individual models and the ensemble of predictions from different classifiers. We further analyze a subtle but fatal issue in the general settings for the self-ensemble model, which causes the deterioration of the weight-ensembled method in the late phases.



## **48. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01287v1)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **49. MIRST-DM: Multi-Instance RST with Drop-Max Layer for Robust Classification of Breast Cancer**

eess.IV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.01674v1)

**Authors**: Shoukun Sun, Min Xian, Aleksandar Vakanski, Hossny Ghanem

**Abstracts**: Robust self-training (RST) can augment the adversarial robustness of image classification models without significantly sacrificing models' generalizability. However, RST and other state-of-the-art defense approaches failed to preserve the generalizability and reproduce their good adversarial robustness on small medical image sets. In this work, we propose the Multi-instance RST with a drop-max layer, namely MIRST-DM, which involves a sequence of iteratively generated adversarial instances during training to learn smoother decision boundaries on small datasets. The proposed drop-max layer eliminates unstable features and helps learn representations that are robust to image perturbations. The proposed approach was validated using a small breast ultrasound dataset with 1,190 images. The results demonstrate that the proposed approach achieves state-of-the-art adversarial robustness against three prevalent attacks.



## **50. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

cs.CV

CVPR 2022 camera ready

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2112.04532v2)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.



