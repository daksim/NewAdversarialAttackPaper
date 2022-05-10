# Latest Adversarial Attack Papers
**update at 2022-05-11 06:31:31**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Energy-bounded Learning for Robust Models of Code**

cs.LG

There are some flaws in our experiments, we would like to fix it and  publish a fixed version again in the very near future

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.11226v2)

**Authors**: Nghi D. Q. Bui, Yijun Yu

**Abstracts**: In programming, learning code representations has a variety of applications, including code classification, code search, comment generation, bug prediction, and so on. Various representations of code in terms of tokens, syntax trees, dependency graphs, code navigation paths, or a combination of their variants have been proposed, however, existing vanilla learning techniques have a major limitation in robustness, i.e., it is easy for the models to make incorrect predictions when the inputs are altered in a subtle way. To enhance the robustness, existing approaches focus on recognizing adversarial samples rather than on the valid samples that fall outside a given distribution, which we refer to as out-of-distribution (OOD) samples. Recognizing such OOD samples is the novel problem investigated in this paper. To this end, we propose to first augment the in=distribution datasets with out-of-distribution samples such that, when trained together, they will enhance the model's robustness. We propose the use of an energy-bounded learning objective function to assign a higher score to in-distribution samples and a lower score to out-of-distribution samples in order to incorporate such out-of-distribution samples into the training process of source code models. In terms of OOD detection and adversarial samples detection, our evaluation results demonstrate a greater robustness for existing source code models to become more accurate at recognizing OOD data while being more resistant to adversarial attacks at the same time. Furthermore, the proposed energy-bounded score outperforms all existing OOD detection scores by a large margin, including the softmax confidence score, the Mahalanobis score, and ODIN.



## **2. Do You Think You Can Hold Me? The Real Challenge of Problem-Space Evasion Attacks**

cs.CR

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04293v1)

**Authors**: Harel Berger, Amit Dvir, Chen Hajaj, Rony Ronen

**Abstracts**: Android malware is a spreading disease in the virtual world. Anti-virus and detection systems continuously undergo patches and updates to defend against these threats. Most of the latest approaches in malware detection use Machine Learning (ML). Against the robustifying effort of detection systems, raise the \emph{evasion attacks}, where an adversary changes its targeted samples so that they are misclassified as benign. This paper considers two kinds of evasion attacks: feature-space and problem-space. \emph{Feature-space} attacks consider an adversary who manipulates ML features to evade the correct classification while minimizing or constraining the total manipulations. \textit{Problem-space} attacks refer to evasion attacks that change the actual sample. Specifically, this paper analyzes the gap between these two types in the Android malware domain. The gap between the two types of evasion attacks is examined via the retraining process of classifiers using each one of the evasion attack types. The experiments show that the gap between these two types of retrained classifiers is dramatic and may increase to 96\%. Retrained classifiers of feature-space evasion attacks have been found to be either less effective or completely ineffective against problem-space evasion attacks. Additionally, exploration of different problem-space evasion attacks shows that retraining of one problem-space evasion attack may be effective against other problem-space evasion attacks.



## **3. Federated Multi-Armed Bandits Under Byzantine Attacks**

cs.LG

13 pages, 15 figures

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04134v1)

**Authors**: Ilker Demirel, Yigit Yildirim, Cem Tekin

**Abstracts**: Multi-armed bandits (MAB) is a simple reinforcement learning model where the learner controls the trade-off between exploration versus exploitation to maximize its cumulative reward. Federated multi-armed bandits (FMAB) is a recently emerging framework where a cohort of learners with heterogeneous local models play a MAB game and communicate their aggregated feedback to a parameter server to learn the global feedback model. Federated learning models are vulnerable to adversarial attacks such as model-update attacks or data poisoning. In this work, we study an FMAB problem in the presence of Byzantine clients who can send false model updates that pose a threat to the learning process. We borrow tools from robust statistics and propose a median-of-means-based estimator: Fed-MoM-UCB, to cope with the Byzantine clients. We show that if the Byzantine clients constitute at most half the cohort, it is possible to incur a cumulative regret on the order of ${\cal O} (\log T)$ with respect to an unavoidable error margin, including the communication cost between the clients and the parameter server. We analyze the interplay between the algorithm parameters, unavoidable error margin, regret, communication cost, and the arms' suboptimality gaps. We demonstrate Fed-MoM-UCB's effectiveness against the baselines in the presence of Byzantine attacks via experiments.



## **4. ResSFL: A Resistance Transfer Framework for Defending Model Inversion Attack in Split Federated Learning**

cs.LG

Accepted to CVPR 2022

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2205.04007v1)

**Authors**: Jingtao Li, Adnan Siraj Rakin, Xing Chen, Zhezhi He, Deliang Fan, Chaitali Chakrabarti

**Abstracts**: This work aims to tackle Model Inversion (MI) attack on Split Federated Learning (SFL). SFL is a recent distributed training scheme where multiple clients send intermediate activations (i.e., feature map), instead of raw data, to a central server. While such a scheme helps reduce the computational load at the client end, it opens itself to reconstruction of raw data from intermediate activation by the server. Existing works on protecting SFL only consider inference and do not handle attacks during training. So we propose ResSFL, a Split Federated Learning Framework that is designed to be MI-resistant during training. It is based on deriving a resistant feature extractor via attacker-aware training, and using this extractor to initialize the client-side model prior to standard SFL training. Such a method helps in reducing the computational complexity due to use of strong inversion model in client-side adversarial training as well as vulnerability of attacks launched in early training epochs. On CIFAR-100 dataset, our proposed framework successfully mitigates MI attack on a VGG-11 model with a high reconstruction Mean-Square-Error of 0.050 compared to 0.005 obtained by the baseline system. The framework achieves 67.5% accuracy (only 1% accuracy drop) with very low computation overhead. Code is released at: https://github.com/zlijingtao/ResSFL.



## **5. Triangle Attack: A Query-efficient Decision-based Adversarial Attack**

cs.CV

10 pages

**SubmitDate**: 2022-05-09    [paper-pdf](http://arxiv.org/pdf/2112.06569v2)

**Authors**: Xiaosen Wang, Zeliang Zhang, Kangheng Tong, Dihong Gong, Kun He, Zhifeng Li, Wei Liu

**Abstracts**: Decision-based attack poses a severe threat to real-world applications since it regards the target model as a black box and only accesses the hard prediction label. Great efforts have been made recently to decrease the number of queries; however, existing decision-based attacks still require thousands of queries in order to generate good quality adversarial examples. In this work, we find that a benign sample, the current and the next adversarial examples could naturally construct a triangle in a subspace for any iterative attacks. Based on the law of sines, we propose a novel Triangle Attack (TA) to optimize the perturbation by utilizing the geometric information that the longer side is always opposite the larger angle in any triangle. However, directly applying such information on the input image is ineffective because it cannot thoroughly explore the neighborhood of the input sample in the high dimensional space. To address this issue, TA optimizes the perturbation in the low frequency space for effective dimensionality reduction owing to the generality of such geometric property. Extensive evaluations on the ImageNet dataset demonstrate that TA achieves a much higher attack success rate within 1,000 queries and needs a much less number of queries to achieve the same attack success rate under various perturbation budgets than existing decision-based attacks. With such high efficiency, we further demonstrate the applicability of TA on real-world API, i.e., Tencent Cloud API.



## **6. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

cs.CR

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2205.03971v1)

**Authors**: Yan Long, Chen Yan, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Personal video conferencing has become the new norm after COVID-19 caused a seismic shift from in-person meetings and phone calls to video conferencing for daily communications and sensitive business. Video leaks participants' on-screen information because eyeglasses and other reflective objects unwittingly expose partial screen contents. Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual information gleamed from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our experimental results and models show it is possible to reconstruct and recognize on-screen text with a height as small as 10 mm with a 720p webcam. We further apply this threat model to web textual content with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution toward 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Our research proposes near-term mitigations, and justifies the importance of following the principle of least privilege for long-term defense against this attack. For privacy-sensitive scenarios, it's further recommended to develop technologies that blur all objects by default, then only unblur what is absolutely necessary to facilitate natural-looking conversations.



## **7. mFI-PSO: A Flexible and Effective Method in Adversarial Image Generation for Deep Neural Networks**

cs.LG

Accepted by 2022 International Joint Conference on Neural Networks  (IJCNN)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2006.03243v3)

**Authors**: Hai Shu, Ronghua Shi, Qiran Jia, Hongtu Zhu, Ziqi Chen

**Abstracts**: Deep neural networks (DNNs) have achieved great success in image classification, but can be very vulnerable to adversarial attacks with small perturbations to images. To improve adversarial image generation for DNNs, we develop a novel method, called mFI-PSO, which utilizes a Manifold-based First-order Influence measure for vulnerable image and pixel selection and the Particle Swarm Optimization for various objective functions. Our mFI-PSO can thus effectively design adversarial images with flexible, customized options on the number of perturbed pixels, the misclassification probability, and the targeted incorrect class. Experiments demonstrate the flexibility and effectiveness of our mFI-PSO in adversarial attacks and its appealing advantages over some popular methods.



## **8. IDSGAN: Generative Adversarial Networks for Attack Generation against Intrusion Detection**

cs.CR

Accepted for publication in the 26th Pacific-Asia Conference on  Knowledge Discovery and Data Mining (PAKDD 2022)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/1809.02077v5)

**Authors**: Zilong Lin, Yong Shi, Zhi Xue

**Abstracts**: As an essential tool in security, the intrusion detection system bears the responsibility of the defense to network attacks performed by malicious traffic. Nowadays, with the help of machine learning algorithms, intrusion detection systems develop rapidly. However, the robustness of this system is questionable when it faces adversarial attacks. For the robustness of detection systems, more potential attack approaches are under research. In this paper, a framework of the generative adversarial networks, called IDSGAN, is proposed to generate the adversarial malicious traffic records aiming to attack intrusion detection systems by deceiving and evading the detection. Given that the internal structure and parameters of the detection system are unknown to attackers, the adversarial attack examples perform the black-box attacks against the detection system. IDSGAN leverages a generator to transform original malicious traffic records into adversarial malicious ones. A discriminator classifies traffic examples and dynamically learns the real-time black-box detection system. More significantly, the restricted modification mechanism is designed for the adversarial generation to preserve original attack functionalities of adversarial traffic records. The effectiveness of the model is indicated by attacking multiple algorithm-based detection models with different attack categories. The robustness is verified by changing the number of the modified features. A comparative experiment with adversarial attack baselines demonstrates the superiority of our model.



## **9. Fingerprinting Deep Neural Networks Globally via Universal Adversarial Perturbations**

cs.CR

Accepted to CVPR 2022 (Oral Presentation)

**SubmitDate**: 2022-05-08    [paper-pdf](http://arxiv.org/pdf/2202.08602v3)

**Authors**: Zirui Peng, Shaofeng Li, Guoxing Chen, Cheng Zhang, Haojin Zhu, Minhui Xue

**Abstracts**: In this paper, we propose a novel and practical mechanism which enables the service provider to verify whether a suspect model is stolen from the victim model via model extraction attacks. Our key insight is that the profile of a DNN model's decision boundary can be uniquely characterized by its Universal Adversarial Perturbations (UAPs). UAPs belong to a low-dimensional subspace and piracy models' subspaces are more consistent with victim model's subspace compared with non-piracy model. Based on this, we propose a UAP fingerprinting method for DNN models and train an encoder via contrastive learning that takes fingerprint as inputs, outputs a similarity score. Extensive studies show that our framework can detect model IP breaches with confidence > 99.99 within only 20 fingerprints of the suspect model. It has good generalizability across different model architectures and is robust against post-modifications on stolen models.



## **10. Poisoning Semi-supervised Federated Learning via Unlabeled Data: Attacks and Defenses**

cs.LG

Updated Version

**SubmitDate**: 2022-05-07    [paper-pdf](http://arxiv.org/pdf/2012.04432v2)

**Authors**: Yi Liu, Xingliang Yuan, Ruihui Zhao, Cong Wang, Dusit Niyato, Yefeng Zheng

**Abstracts**: Semi-supervised Federated Learning (SSFL) has recently drawn much attention due to its practical consideration, i.e., the clients may only have unlabeled data. In practice, these SSFL systems implement semi-supervised training by assigning a "guessed" label to the unlabeled data near the labeled data to convert the unsupervised problem into a fully supervised problem. However, the inherent properties of such semi-supervised training techniques create a new attack surface. In this paper, we discover and reveal a simple yet powerful poisoning attack against SSFL. Our attack utilizes the natural characteristic of semi-supervised learning to cause the model to be poisoned by poisoning unlabeled data. Specifically, the adversary just needs to insert a small number of maliciously crafted unlabeled samples (e.g., only 0.1\% of the dataset) to infect model performance and misclassification. Extensive case studies have shown that our attacks are effective on different datasets and common semi-supervised learning methods. To mitigate the attacks, we propose a defense, i.e., a minimax optimization-based client selection strategy, to enable the server to select the clients who hold the correct label information and high-quality updates. Our defense further employs a quality-based aggregation rule to strengthen the contributions of the selected updates. Evaluations under different attack conditions show that the proposed defense can well alleviate such unlabeled poisoning attacks. Our study unveils the vulnerability of SSFL to unlabeled poisoning attacks and provides the community with potential defense methods.



## **11. Using cyber threat intelligence to support adversary understanding applied to the Russia-Ukraine conflict**

cs.CR

in Spanish language

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03469v1)

**Authors**: Oscar Sandoval Carlos

**Abstracts**: In military organizations, Cyber Threat Intelligence (CTI) supports cyberspace operations by providing the commander with essential information about the adversary, their capabilities and objectives as they operate through cyberspace. This paper, combines CTI with the MITRE ATT&CK framework in order to establish an adversary profile. In addition, it identifies the characteristics of the attack phase by analyzing the WhisperGate operation that occurred in Ukraine in January 2022, and suggests the minimum essential measures for defense.



## **12. Subverting Fair Image Search with Generative Adversarial Perturbations**

cs.LG

Accepted as a full paper at the 2022 ACM Conference on Fairness,  Accountability, and Transparency (FAccT 22)

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02414v2)

**Authors**: Avijit Ghosh, Matthew Jagielski, Christo Wilson

**Abstracts**: In this work we explore the intersection fairness and robustness in the context of ranking: when a ranking model has been calibrated to achieve some definition of fairness, is it possible for an external adversary to make the ranking model behave unfairly without having access to the model or training data? To investigate this question, we present a case study in which we develop and then attack a state-of-the-art, fairness-aware image search engine using images that have been maliciously modified using a Generative Adversarial Perturbation (GAP) model. These perturbations attempt to cause the fair re-ranking algorithm to unfairly boost the rank of images containing people from an adversary-selected subpopulation.   We present results from extensive experiments demonstrating that our attacks can successfully confer significant unfair advantage to people from the majority class relative to fairly-ranked baseline search results. We demonstrate that our attacks are robust across a number of variables, that they have close to zero impact on the relevance of search results, and that they succeed under a strict threat model. Our findings highlight the danger of deploying fair machine learning algorithms in-the-wild when (1) the data necessary to achieve fairness may be adversarially manipulated, and (2) the models themselves are not robust against attacks.



## **13. Leveraging strategic connection migration-powered traffic splitting for privacy**

cs.CR

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.03326v1)

**Authors**: Mona Wang, Anunay Kulshrestha, Liang Wang, Prateek Mittal

**Abstracts**: Network-level adversaries have developed increasingly sophisticated techniques to surveil and control users' network traffic. In this paper, we exploit our observation that many encrypted protocol connections are no longer tied to device IP address (e.g., the connection migration feature in QUIC, or IP roaming in WireGuard and Mosh), due to the need for performance in a mobile-first world. We design and implement a novel framework, Connection Migration Powered Splitting (CoMPS), that utilizes these performance features for enhancing user privacy. With CoMPS, we can split traffic mid-session across network paths and heterogeneous network protocols. Such traffic splitting mitigates the ability of a network-level adversary to perform traffic analysis attacks by limiting the amount of traffic they can observe. We use CoMPS to construct a website fingerprinting defense that is resilient against traffic analysis attacks by a powerful adaptive adversary in the open-world setting. We evaluate our system using both simulated splitting data and real-world traffic that is actively split using CoMPS. In our real-world experiments, CoMPS reduces the precision and recall of VarCNN to 29.9% and 36.7% respectively in the open-world setting with 100 monitored classes. CoMPS is not only immediately deployable with any unaltered server that supports connection migration, but also incurs little overhead, decreasing throughput by only 5-20%.



## **14. Adversarial Classification under Gaussian Mechanism: Calibrating the Attack to Sensitivity**

cs.IT

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2201.09751v3)

**Authors**: Ayse Unsal, Melek Onen

**Abstracts**: This work studies anomaly detection under differential privacy (DP) with Gaussian perturbation using both statistical and information-theoretic tools. In our setting, the adversary aims to modify the content of a statistical dataset by inserting additional data without being detected by using the DP guarantee to her own benefit. To this end, we characterize information-theoretic and statistical thresholds for the first and second-order statistics of the adversary's attack, which balances the privacy budget and the impact of the attack in order to remain undetected. Additionally, we introduce a new privacy metric based on Chernoff information for classifying adversaries under differential privacy as a stronger alternative to $(\epsilon, \delta)-$ and Kullback-Leibler DP for the Gaussian mechanism. Analytical results are supported by numerical evaluations.



## **15. Learning Optimal Propagation for Graph Neural Networks**

cs.LG

7 pages, 3 figures

**SubmitDate**: 2022-05-06    [paper-pdf](http://arxiv.org/pdf/2205.02998v1)

**Authors**: Beidi Zhao, Boxin Du, Zhe Xu, Liangyue Li, Hanghang Tong

**Abstracts**: Graph Neural Networks (GNNs) have achieved tremendous success in a variety of real-world applications by relying on the fixed graph data as input. However, the initial input graph might not be optimal in terms of specific downstream tasks, because of information scarcity, noise, adversarial attacks, or discrepancies between the distribution in graph topology, features, and groundtruth labels. In this paper, we propose a bi-level optimization-based approach for learning the optimal graph structure via directly learning the Personalized PageRank propagation matrix as well as the downstream semi-supervised node classification simultaneously. We also explore a low-rank approximation model for further reducing the time complexity. Empirical evaluations show the superior efficacy and robustness of the proposed model over all baseline methods.



## **16. Privacy-from-Birth: Protecting Sensed Data from Malicious Sensors with VERSA**

cs.CR

13 pages paper and 4 pages appendix. To be published at 2022 IEEE  Symposium on Security and Privacy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02963v1)

**Authors**: Ivan De Oliveira Nunes, Seoyeon Hwang, Sashidhar Jakkamsetti, Gene Tsudik

**Abstracts**: There are many well-known techniques to secure sensed data in IoT/CPS systems, e.g., by authenticating communication end-points, encrypting data before transmission, and obfuscating traffic patterns. Such techniques protect sensed data from external adversaries while assuming that the sensing device itself is secure. Meanwhile, both the scale and frequency of IoT-focused attacks are growing. This prompts a natural question: how to protect sensed data even if all software on the device is compromised? Ideally, in order to achieve this, sensed data must be protected from its genesis, i.e., from the time when a physical analog quantity is converted into its digital counterpart and becomes accessible to software. We refer to this property as PfB: Privacy-from-Birth.   In this work, we formalize PfB and design Verified Remote Sensing Authorization (VERSA) -- a provably secure and formally verified architecture guaranteeing that only correct execution of expected and explicitly authorized software can access and manipulate sensing interfaces, specifically, General Purpose Input/Output (GPIO), which is the usual boundary between analog and digital worlds on IoT devices. This guarantee is obtained with minimal hardware support and holds even if all device software is compromised. VERSA ensures that malware can neither gain access to sensed data on the GPIO-mapped memory nor obtain any trace thereof. VERSA is formally verified and its open-sourced implementation targets resource-constrained IoT edge devices, commonly used for sensing. Experimental results show that PfB is both achievable and affordable for such devices.



## **17. Transferring Adversarial Robustness Through Robust Representation Matching**

cs.LG

To appear at USENIX Security '22. Updated version with artifact  evaluation badges and appendix

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2202.09994v2)

**Authors**: Pratik Vaishnavi, Kevin Eykholt, Amir Rahmati

**Abstracts**: With the widespread use of machine learning, concerns over its security and reliability have become prevalent. As such, many have developed defenses to harden neural networks against adversarial examples, imperceptibly perturbed inputs that are reliably misclassified. Adversarial training in which adversarial examples are generated and used during training is one of the few known defenses able to reliably withstand such attacks against neural networks. However, adversarial training imposes a significant training overhead and scales poorly with model complexity and input dimension. In this paper, we propose Robust Representation Matching (RRM), a low-cost method to transfer the robustness of an adversarially trained model to a new model being trained for the same task irrespective of architectural differences. Inspired by student-teacher learning, our method introduces a novel training loss that encourages the student to learn the teacher's robust representations. Compared to prior works, RRM is superior with respect to both model performance and adversarial training time. On CIFAR-10, RRM trains a robust model $\sim 1.8\times$ faster than the state-of-the-art. Furthermore, RRM remains effective on higher-dimensional datasets. On Restricted-ImageNet, RRM trains a ResNet50 model $\sim 18\times$ faster than standard adversarial training.



## **18. Can collaborative learning be private, robust and scalable?**

cs.LG

Submitted to TPDP 2022

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02652v1)

**Authors**: Dmitrii Usynin, Helena Klause, Daniel Rueckert, Georgios Kaissis

**Abstracts**: We investigate the effectiveness of combining differential privacy, model compression and adversarial training to improve the robustness of models against adversarial samples in train- and inference-time attacks. We explore the applications of these techniques as well as their combinations to determine which method performs best, without a significant utility trade-off. Our investigation provides a practical overview of various methods that allow one to achieve a competitive model performance, a significant reduction in model's size and an improved empirical adversarial robustness without a severe performance degradation.



## **19. Holistic Approach to Measure Sample-level Adversarial Vulnerability and its Utility in Building Trustworthy Systems**

cs.CV

Accepted in CVPR Workshop 2022 on Human-centered Intelligent  Services: Safe and Trustworthy

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02604v1)

**Authors**: Gaurav Kumar Nayak, Ruchit Rawal, Rohit Lal, Himanshu Patil, Anirban Chakraborty

**Abstracts**: Adversarial attack perturbs an image with an imperceptible noise, leading to incorrect model prediction. Recently, a few works showed inherent bias associated with such attack (robustness bias), where certain subgroups in a dataset (e.g. based on class, gender, etc.) are less robust than others. This bias not only persists even after adversarial training, but often results in severe performance discrepancies across these subgroups. Existing works characterize the subgroup's robustness bias by only checking individual sample's proximity to the decision boundary. In this work, we argue that this measure alone is not sufficient and validate our argument via extensive experimental analysis. It has been observed that adversarial attacks often corrupt the high-frequency components of the input image. We, therefore, propose a holistic approach for quantifying adversarial vulnerability of a sample by combining these different perspectives, i.e., degree of model's reliance on high-frequency features and the (conventional) sample-distance to the decision boundary. We demonstrate that by reliably estimating adversarial vulnerability at the sample level using the proposed holistic metric, it is possible to develop a trustworthy system where humans can be alerted about the incoming samples that are highly likely to be misclassified at test time. This is achieved with better precision when our holistic metric is used over individual measures. To further corroborate the utility of the proposed holistic approach, we perform knowledge distillation in a limited-sample setting. We observe that the student network trained with the subset of samples selected using our combined metric performs better than both the competing baselines, viz., where samples are selected randomly or based on their distances to the decision boundary.



## **20. Resilience of Bayesian Layer-Wise Explanations under Adversarial Attacks**

cs.LG

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2102.11010v3)

**Authors**: Ginevra Carbone, Guido Sanguinetti, Luca Bortolussi

**Abstracts**: We consider the problem of the stability of saliency-based explanations of Neural Network predictions under adversarial attacks in a classification task. Saliency interpretations of deterministic Neural Networks are remarkably brittle even when the attacks fail, i.e. for attacks that do not change the classification label. We empirically show that interpretations provided by Bayesian Neural Networks are considerably more stable under adversarial perturbations of the inputs and even under direct attacks to the explanations. By leveraging recent results, we also provide a theoretical explanation of this result in terms of the geometry of the data manifold. Additionally, we discuss the stability of the interpretations of high level representations of the inputs in the internal layers of a Network. Our results demonstrate that Bayesian methods, in addition to being more robust to adversarial attacks, have the potential to provide more stable and interpretable assessments of Neural Network predictions.



## **21. Robust Conversational Agents against Imperceptible Toxicity Triggers**

cs.CL

**SubmitDate**: 2022-05-05    [paper-pdf](http://arxiv.org/pdf/2205.02392v1)

**Authors**: Ninareh Mehrabi, Ahmad Beirami, Fred Morstatter, Aram Galstyan

**Abstracts**: Warning: this paper contains content that maybe offensive or upsetting. Recent research in Natural Language Processing (NLP) has advanced the development of various toxicity detection models with the intention of identifying and mitigating toxic language from existing systems. Despite the abundance of research in this area, less attention has been given to adversarial attacks that force the system to generate toxic language and the defense against them. Existing work to generate such attacks is either based on human-generated attacks which is costly and not scalable or, in case of automatic attacks, the attack vector does not conform to human-like language, which can be detected using a language model loss. In this work, we propose attacks against conversational agents that are imperceptible, i.e., they fit the conversation in terms of coherency, relevancy, and fluency, while they are effective and scalable, i.e., they can automatically trigger the system into generating toxic language. We then propose a defense mechanism against such attacks which not only mitigates the attack but also attempts to maintain the conversational flow. Through automatic and human evaluations, we show that our defense is effective at avoiding toxic language generation even against imperceptible toxicity triggers while the generated language fits the conversation in terms of coherency and relevancy. Lastly, we establish the generalizability of such a defense mechanism on language generation models beyond conversational agents.



## **22. Zero Day Threat Detection Using Graph and Flow Based Security Telemetry**

cs.CR

11 pages, 6 figures, submitting to NeurIPS 2022

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02298v1)

**Authors**: Christopher Redino, Dhruv Nandakumar, Robert Schiller, Kevin Choi, Abdul Rahman, Edward Bowen, Matthew Weeks, Aaron Shaha, Joe Nehila

**Abstracts**: Zero Day Threats (ZDT) are novel methods used by malicious actors to attack and exploit information technology (IT) networks or infrastructure. In the past few years, the number of these threats has been increasing at an alarming rate and have been costing organizations millions of dollars to remediate. The increasing expansion of network attack surfaces and the exponentially growing number of assets on these networks necessitate the need for a robust AI-based Zero Day Threat detection model that can quickly analyze petabyte-scale data for potentially malicious and novel activity. In this paper, the authors introduce a deep learning based approach to Zero Day Threat detection that can generalize, scale, and effectively identify threats in near real-time. The methodology utilizes network flow telemetry augmented with asset-level graph features, which are passed through a dual-autoencoder structure for anomaly and novelty detection respectively. The models have been trained and tested on four large scale datasets that are representative of real-world organizational networks and they produce strong results with high precision and recall values. The models provide a novel methodology to detect complex threats with low false-positive rates that allow security operators to avoid alert fatigue while drastically reducing their mean time to response with near-real-time detection. Furthermore, the authors also provide a novel, labelled, cyber attack dataset generated from adversarial activity that can be used for validation or training of other models. With this paper, the authors' overarching goal is to provide a novel architecture and training methodology for cyber anomaly detectors that can generalize to multiple IT networks with minimal to no retraining while still maintaining strong performance.



## **23. Adversarial Training for High-Stakes Reliability**

cs.LG

31 pages, 6 figures, small tweak

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.01663v2)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **24. Rethinking Classifier And Adversarial Attack**

cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02743v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Various defense models have been proposed to resist adversarial attack algorithms, but existing adversarial robustness evaluation methods always overestimate the adversarial robustness of these models (i.e. not approaching the lower bound of robustness). To solve this problem, this paper first uses the Decouple Space method to divide the classifier into two parts: non-linear and linear. On this basis, this paper defines the representation vector of original example (and its space, i.e., the representation space) and uses Absolute Classification Boundaries Initialization (ACBI) iterative optimization to obtain a better attack starting point (i.e. attacking from this point can approach the lower bound of robustness faster). Particularly, this paper apply ACBI to nearly 50 widely-used defense models (including 8 architectures). Experimental results show that ACBI achieves lower robust accuracy in all cases.



## **25. Based-CE white-box adversarial attack will not work using super-fitting**

cs.LG

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2205.02741v1)

**Authors**: Youhuan Yang, Lei Sun, Leyu Dai, Song Guo, Xiuqing Mao, Xiaoqin Wang, Bayi Xu

**Abstracts**: Deep Neural Networks (DNN) are widely used in various fields due to their powerful performance, but recent studies have shown that deep learning models are vulnerable to adversarial attacks-by adding a slight perturbation to the input, the model will get wrong results. It is especially dangerous for some systems with high security requirements, so this paper proposes a new defense method by using the model super-fitting status. Model's adversarial robustness (i.e., the accuracry under adversarial attack) has been greatly improved in this status. This paper mathematically proves the effectiveness of super-fitting, and proposes a method to make the model reach this status quickly-minimaze unrelated categories scores (MUCS). Theoretically, super-fitting can resist any existing (even future) Based on CE white-box adversarial attack. In addition, this paper uses a variety of powerful attack algorithms to evaluate the adversarial robustness of super-fitting and other nearly 50 defense models from recent conferences. The experimental results show that super-fitting method in this paper can make the trained model obtain the highest adversarial performance robustness.



## **26. Few-Shot Backdoor Attacks on Visual Object Tracking**

cs.CV

This work is accepted by the ICLR 2022. The first two authors  contributed equally to this work. In this version, we fix some typos and  errors contained in the last one. 21 pages

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2201.13178v2)

**Authors**: Yiming Li, Haoxiang Zhong, Xingjun Ma, Yong Jiang, Shu-Tao Xia

**Abstracts**: Visual object tracking (VOT) has been widely adopted in mission-critical applications, such as autonomous driving and intelligent surveillance systems. In current practice, third-party resources such as datasets, backbone networks, and training platforms are frequently used to train high-performance VOT models. Whilst these resources bring certain convenience, they also introduce new security threats into VOT models. In this paper, we reveal such a threat where an adversary can easily implant hidden backdoors into VOT models by tempering with the training process. Specifically, we propose a simple yet effective few-shot backdoor attack (FSBA) that optimizes two losses alternately: 1) a \emph{feature loss} defined in the hidden feature space, and 2) the standard \emph{tracking loss}. We show that, once the backdoor is embedded into the target model by our FSBA, it can trick the model to lose track of specific objects even when the \emph{trigger} only appears in one or a few frames. We examine our attack in both digital and physical-world settings and show that it can significantly degrade the performance of state-of-the-art VOT trackers. We also show that our attack is resistant to potential defenses, highlighting the vulnerability of VOT models to potential backdoor attacks.



## **27. Authentication Attacks on Projection-based Cancelable Biometric Schemes (long version)**

cs.CR

arXiv admin note: text overlap with arXiv:1910.01389 by other authors

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2110.15163v3)

**Authors**: Axel Durbet, Pascal Lafourcade, Denis Migdal, Kevin Thiry-Atighehchi, Paul-Marie Grollemund

**Abstracts**: Cancelable biometric schemes aim at generating secure biometric templates by combining user specific tokens, such as password, stored secret or salt, along with biometric data. This type of transformation is constructed as a composition of a biometric transformation with a feature extraction algorithm. The security requirements of cancelable biometric schemes concern the irreversibility, unlinkability and revocability of templates, without losing in accuracy of comparison. While several schemes were recently attacked regarding these requirements, full reversibility of such a composition in order to produce colliding biometric characteristics, and specifically presentation attacks, were never demonstrated to the best of our knowledge. In this paper, we formalize these attacks for a traditional cancelable scheme with the help of integer linear programming (ILP) and quadratically constrained quadratic programming (QCQP). Solving these optimization problems allows an adversary to slightly alter its fingerprint image in order to impersonate any individual. Moreover, in an even more severe scenario, it is possible to simultaneously impersonate several individuals.



## **28. AdaptOver: Adaptive Overshadowing Attacks in Cellular Networks**

cs.CR

This version introduces uplink overshadowing

**SubmitDate**: 2022-05-04    [paper-pdf](http://arxiv.org/pdf/2106.05039v2)

**Authors**: Simon Erni, Martin Kotuliak, Patrick Leu, Marc Röschlin, Srdjan Čapkun

**Abstracts**: In cellular networks, attacks on the communication link between a mobile device and the core network significantly impact privacy and availability. Up until now, fake base stations have been required to execute such attacks. Since they require a continuously high output power to attract victims, they are limited in range and can be easily detected both by operators and dedicated apps on users' smartphones.   This paper introduces AdaptOver -- a MITM attack system designed for cellular networks, specifically for LTE and 5G-NSA. AdaptOver allows an adversary to decode, overshadow (replace) and inject arbitrary messages over the air in either direction between the network and the mobile device. Using overshadowing, AdaptOver can cause a persistent ($\geq$ 12h) DoS or a privacy leak by triggering a UE to transmit its persistent identifier (IMSI) in plain text. These attacks can be launched against all users within a cell or specifically target a victim based on its phone number.   We implement AdaptOver using a software-defined radio and a low-cost amplification setup. We demonstrate the effects and practicality of the attacks on a live operational LTE and 5G-NSA network with a wide range of smartphones. Our experiments show that AdaptOver can launch an attack on a victim more than 3.8km away from the attacker. Given its practicability and efficiency, AdaptOver shows that existing countermeasures that are focused on fake base stations are no longer sufficient, marking a paradigm shift for designing security mechanisms in cellular networks.



## **29. Can Rationalization Improve Robustness?**

cs.CL

Accepted to NAACL 2022; The code is available at  https://github.com/princeton-nlp/rationale-robustness

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2204.11790v2)

**Authors**: Howard Chen, Jacqueline He, Karthik Narasimhan, Danqi Chen

**Abstracts**: A growing line of work has investigated the development of neural NLP models that can produce rationales--subsets of input that can explain their model predictions. In this paper, we ask whether such rationale models can also provide robustness to adversarial attacks in addition to their interpretable nature. Since these models need to first generate rationales ("rationalizer") before making predictions ("predictor"), they have the potential to ignore noise or adversarially added text by simply masking it out of the generated rationale. To this end, we systematically generate various types of 'AddText' attacks for both token and sentence-level rationalization tasks, and perform an extensive empirical evaluation of state-of-the-art rationale models across five different tasks. Our experiments reveal that the rationale models show the promise to improve robustness, while they struggle in certain scenarios--when the rationalizer is sensitive to positional bias or lexical choices of attack text. Further, leveraging human rationale as supervision does not always translate to better performance. Our study is a first step towards exploring the interplay between interpretability and robustness in the rationalize-then-predict framework.



## **30. Don't sweat the small stuff, classify the rest: Sample Shielding to protect text classifiers against adversarial attacks**

cs.CL

9 pages, 8 figures, Accepted to NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01714v1)

**Authors**: Jonathan Rusert, Padmini Srinivasan

**Abstracts**: Deep learning (DL) is being used extensively for text classification. However, researchers have demonstrated the vulnerability of such classifiers to adversarial attacks. Attackers modify the text in a way which misleads the classifier while keeping the original meaning close to intact. State-of-the-art (SOTA) attack algorithms follow the general principle of making minimal changes to the text so as to not jeopardize semantics. Taking advantage of this we propose a novel and intuitive defense strategy called Sample Shielding. It is attacker and classifier agnostic, does not require any reconfiguration of the classifier or external resources and is simple to implement. Essentially, we sample subsets of the input text, classify them and summarize these into a final decision. We shield three popular DL text classifiers with Sample Shielding, test their resilience against four SOTA attackers across three datasets in a realistic threat setting. Even when given the advantage of knowing about our shielding strategy the adversary's attack success rate is <=10% with only one exception and often < 5%. Additionally, Sample Shielding maintains near original accuracy when applied to original texts. Crucially, we show that the `make minimal changes' approach of SOTA attackers leads to critical vulnerabilities that can be defended against with an intuitive sampling strategy.



## **31. A Unified Framework for Adversarial Attack and Defense in Constrained Feature Space**

cs.AI

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2112.01156v2)

**Authors**: Thibault Simonetto, Salijona Dyrmishi, Salah Ghamizi, Maxime Cordy, Yves Le Traon

**Abstracts**: The generation of feasible adversarial examples is necessary for properly assessing models that work in constrained feature space. However, it remains a challenging task to enforce constraints into attacks that were designed for computer vision. We propose a unified framework to generate feasible adversarial examples that satisfy given domain constraints. Our framework can handle both linear and non-linear constraints. We instantiate our framework into two algorithms: a gradient-based attack that introduces constraints in the loss function to maximize, and a multi-objective search algorithm that aims for misclassification, perturbation minimization, and constraint satisfaction. We show that our approach is effective in four different domains, with a success rate of up to 100%, where state-of-the-art attacks fail to generate a single feasible example. In addition to adversarial retraining, we propose to introduce engineered non-convex constraints to improve model adversarial robustness. We demonstrate that this new defense is as effective as adversarial retraining. Our framework forms the starting point for research on constrained adversarial attacks and provides relevant baselines and datasets that future research can exploit.



## **32. On the uncertainty principle of neural networks**

cs.LG

8 pages, 8 figures

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01493v1)

**Authors**: Jun-Jie Zhang, Dong-Xiao Zhang, Jian-Nan Chen, Long-Gang Pang

**Abstracts**: Despite the successes in many fields, it is found that neural networks are vulnerability and difficult to be both accurate and robust (robust means that the prediction of the trained network stays unchanged for inputs with non-random perturbations introduced by adversarial attacks). Various empirical and analytic studies have suggested that there is more or less a trade-off between the accuracy and robustness of neural networks. If the trade-off is inherent, applications based on the neural networks are vulnerable with untrustworthy predictions. It is then essential to ask whether the trade-off is an inherent property or not. Here, we show that the accuracy-robustness trade-off is an intrinsic property whose underlying mechanism is deeply related to the uncertainty principle in quantum mechanics. We find that for a neural network to be both accurate and robust, it needs to resolve the features of the two conjugated parts $x$ (the inputs) and $\Delta$ (the derivatives of the normalized loss function $J$ with respect to $x$), respectively. Analogous to the position-momentum conjugation in quantum mechanics, we show that the inputs and their conjugates cannot be resolved by a neural network simultaneously.



## **33. Self-Ensemble Adversarial Training for Improved Robustness**

cs.LG

18 pages, 3 figures, ICLR 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2203.09678v2)

**Authors**: Hongjun Wang, Yisen Wang

**Abstracts**: Due to numerous breakthroughs in real-world applications brought by machine intelligence, deep neural networks (DNNs) are widely employed in critical applications. However, predictions of DNNs are easily manipulated with imperceptible adversarial perturbations, which impedes the further deployment of DNNs and may result in profound security and privacy implications. By incorporating adversarial samples into the training data pool, adversarial training is the strongest principled strategy against various adversarial attacks among all sorts of defense methods. Recent works mainly focus on developing new loss functions or regularizers, attempting to find the unique optimal point in the weight space. But none of them taps the potentials of classifiers obtained from standard adversarial training, especially states on the searching trajectory of training. In this work, we are dedicated to the weight states of models through the training process and devise a simple but powerful \emph{Self-Ensemble Adversarial Training} (SEAT) method for yielding a robust classifier by averaging weights of history models. This considerably improves the robustness of the target model against several well known adversarial attacks, even merely utilizing the naive cross-entropy loss to supervise. We also discuss the relationship between the ensemble of predictions from different adversarially trained models and the prediction of weight-ensembled models, as well as provide theoretical and empirical evidence that the proposed self-ensemble method provides a smoother loss landscape and better robustness than both individual models and the ensemble of predictions from different classifiers. We further analyze a subtle but fatal issue in the general settings for the self-ensemble model, which causes the deterioration of the weight-ensembled method in the late phases.



## **34. SemAttack: Natural Textual Attacks via Different Semantic Spaces**

cs.CL

Published at Findings of NAACL 2022

**SubmitDate**: 2022-05-03    [paper-pdf](http://arxiv.org/pdf/2205.01287v1)

**Authors**: Boxin Wang, Chejian Xu, Xiangyu Liu, Yu Cheng, Bo Li

**Abstracts**: Recent studies show that pre-trained language models (LMs) are vulnerable to textual adversarial attacks. However, existing attack methods either suffer from low attack success rates or fail to search efficiently in the exponentially large perturbation space. We propose an efficient and effective framework SemAttack to generate natural adversarial text by constructing different semantic perturbation functions. In particular, SemAttack optimizes the generated perturbations constrained on generic semantic spaces, including typo space, knowledge space (e.g., WordNet), contextualized semantic space (e.g., the embedding space of BERT clusterings), or the combination of these spaces. Thus, the generated adversarial texts are more semantically close to the original inputs. Extensive experiments reveal that state-of-the-art (SOTA) large-scale LMs (e.g., DeBERTa-v2) and defense strategies (e.g., FreeLB) are still vulnerable to SemAttack. We further demonstrate that SemAttack is general and able to generate natural adversarial texts for different languages (e.g., English and Chinese) with high attack success rates. Human evaluations also confirm that our generated adversarial texts are natural and barely affect human performance. Our code is publicly available at https://github.com/AI-secure/SemAttack.



## **35. MIRST-DM: Multi-Instance RST with Drop-Max Layer for Robust Classification of Breast Cancer**

eess.IV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.01674v1)

**Authors**: Shoukun Sun, Min Xian, Aleksandar Vakanski, Hossny Ghanem

**Abstracts**: Robust self-training (RST) can augment the adversarial robustness of image classification models without significantly sacrificing models' generalizability. However, RST and other state-of-the-art defense approaches failed to preserve the generalizability and reproduce their good adversarial robustness on small medical image sets. In this work, we propose the Multi-instance RST with a drop-max layer, namely MIRST-DM, which involves a sequence of iteratively generated adversarial instances during training to learn smoother decision boundaries on small datasets. The proposed drop-max layer eliminates unstable features and helps learn representations that are robust to image perturbations. The proposed approach was validated using a small breast ultrasound dataset with 1,190 images. The results demonstrate that the proposed approach achieves state-of-the-art adversarial robustness against three prevalent attacks.



## **36. Segment and Complete: Defending Object Detectors against Adversarial Patch Attacks with Robust Patch Detection**

cs.CV

CVPR 2022 camera ready

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2112.04532v2)

**Authors**: Jiang Liu, Alexander Levine, Chun Pong Lau, Rama Chellappa, Soheil Feizi

**Abstracts**: Object detection plays a key role in many security-critical systems. Adversarial patch attacks, which are easy to implement in the physical world, pose a serious threat to state-of-the-art object detectors. Developing reliable defenses for object detectors against patch attacks is critical but severely understudied. In this paper, we propose Segment and Complete defense (SAC), a general framework for defending object detectors against patch attacks through detection and removal of adversarial patches. We first train a patch segmenter that outputs patch masks which provide pixel-level localization of adversarial patches. We then propose a self adversarial training algorithm to robustify the patch segmenter. In addition, we design a robust shape completion algorithm, which is guaranteed to remove the entire patch from the images if the outputs of the patch segmenter are within a certain Hamming distance of the ground-truth patch masks. Our experiments on COCO and xView datasets demonstrate that SAC achieves superior robustness even under strong adaptive attacks with no reduction in performance on clean images, and generalizes well to unseen patch shapes, attack budgets, and unseen attack methods. Furthermore, we present the APRICOT-Mask dataset, which augments the APRICOT dataset with pixel-level annotations of adversarial patches. We show SAC can significantly reduce the targeted attack success rate of physical patch attacks. Our code is available at https://github.com/joellliu/SegmentAndComplete.



## **37. Defending Against Advanced Persistent Threats using Game-Theory**

cs.CR

preprint of a correction to the article with the same name, published  with PLOS ONE, and currently under review

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00956v1)

**Authors**: Stefan Rass, Sandra König, Stefan Schauer

**Abstracts**: Advanced persistent threats (APT) combine a variety of different attack forms ranging from social engineering to technical exploits. The diversity and usual stealthiness of APT turns them into a central problem of contemporary practical system security, since information on attacks, the current system status or the attacker's incentives is often vague, uncertain and in many cases even unavailable. Game theory is a natural approach to model the conflict between the attacker and the defender, and this work investigates a generalized class of matrix games as a risk mitigation tool for an APT defense. Unlike standard game and decision theory, our model is tailored to capture and handle the full uncertainty that is immanent to APT, such as disagreement among qualitative expert risk assessments, unknown adversarial incentives and uncertainty about the current system state (in terms of how deeply the attacker may have penetrated into the system's protective shells already). Practically, game-theoretic APT models can be derived straightforwardly from topological vulnerability analysis, together with risk assessments as they are done in common risk management standards like the ISO 31000 family. Theoretically, these models come with different properties than classical game theoretic models, whose technical solution presented in this work may be of independent interest.



## **38. BERTops: Studying BERT Representations under a Topological Lens**

cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00953v1)

**Authors**: Jatin Chauhan, Manohar Kaul

**Abstracts**: Proposing scoring functions to effectively understand, analyze and learn various properties of high dimensional hidden representations of large-scale transformer models like BERT can be a challenging task. In this work, we explore a new direction by studying the topological features of BERT hidden representations using persistent homology (PH). We propose a novel scoring function named "persistence scoring function (PSF)" which: (i) accurately captures the homology of the high-dimensional hidden representations and correlates well with the test set accuracy of a wide range of datasets and outperforms existing scoring metrics, (ii) captures interesting post fine-tuning "per-class" level properties from both qualitative and quantitative viewpoints, (iii) is more stable to perturbations as compared to the baseline functions, which makes it a very robust proxy, and (iv) finally, also serves as a predictor of the attack success rates for a wide category of black-box and white-box adversarial attack methods. Our extensive correlation experiments demonstrate the practical utility of PSF on various NLP tasks relevant to BERT.



## **39. Revisiting Gaussian Neurons for Online Clustering with Unknown Number of Clusters**

cs.LG

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00920v1)

**Authors**: Ole Christian Eidheim

**Abstracts**: Despite the recent success of artificial neural networks, more biologically plausible learning methods may be needed to resolve the weaknesses of backpropagation trained models such as catastrophic forgetting and adversarial attacks. A novel local learning rule is presented that performs online clustering with a maximum limit of the number of cluster to be found rather than a fixed cluster count. Instead of using orthogonal weight or output activation constraints, activation sparsity is achieved by mutual repulsion of lateral Gaussian neurons ensuring that multiple neuron centers cannot occupy the same location in the input domain. An update method is also presented for adjusting the widths of the Gaussian neurons in cases where the data samples can be represented by means and variances. The algorithms were applied on the MNIST and CIFAR-10 datasets to create filters capturing the input patterns of pixel patches of various sizes. The experimental results demonstrate stability in the learned parameters across a large number of training samples.



## **40. Deep-Attack over the Deep Reinforcement Learning**

cs.LG

Accepted to Knowledge-Based Systems

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00807v1)

**Authors**: Yang Li, Quan Pan, Erik Cambria

**Abstracts**: Recent adversarial attack developments have made reinforcement learning more vulnerable, and different approaches exist to deploy attacks against it, where the key is how to choose the right timing of the attack. Some work tries to design an attack evaluation function to select critical points that will be attacked if the value is greater than a certain threshold. This approach makes it difficult to find the right place to deploy an attack without considering the long-term impact. In addition, there is a lack of appropriate indicators of assessment during attacks. To make the attacks more intelligent as well as to remedy the existing problems, we propose the reinforcement learning-based attacking framework by considering the effectiveness and stealthy spontaneously, while we also propose a new metric to evaluate the performance of the attack model in these two aspects. Experimental results show the effectiveness of our proposed model and the goodness of our proposed evaluation metric. Furthermore, we validate the transferability of the model, and also its robustness under the adversarial training.



## **41. Enhancing Adversarial Training with Feature Separability**

cs.CV

10 pages

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00637v1)

**Authors**: Yaxin Li, Xiaorui Liu, Han Xu, Wentao Wang, Jiliang Tang

**Abstracts**: Deep Neural Network (DNN) are vulnerable to adversarial attacks. As a countermeasure, adversarial training aims to achieve robustness based on the min-max optimization problem and it has shown to be one of the most effective defense strategies. However, in this work, we found that compared with natural training, adversarial training fails to learn better feature representations for either clean or adversarial samples, which can be one reason why adversarial training tends to have severe overfitting issues and less satisfied generalize performance. Specifically, we observe two major shortcomings of the features learned by existing adversarial training methods:(1) low intra-class feature similarity; and (2) conservative inter-classes feature variance. To overcome these shortcomings, we introduce a new concept of adversarial training graph (ATG) with which the proposed adversarial training with feature separability (ATFS) enables to coherently boost the intra-class feature similarity and increase inter-class feature variance. Through comprehensive experiments, we demonstrate that the proposed ATFS framework significantly improves both clean and robust performance.



## **42. Robust Fine-tuning via Perturbation and Interpolation from In-batch Instances**

cs.CL

IJCAI-ECAI 2022 (the 31st International Joint Conference on  Artificial Intelligence and the 25th European Conference on Artificial  Intelligence)

**SubmitDate**: 2022-05-02    [paper-pdf](http://arxiv.org/pdf/2205.00633v1)

**Authors**: Shoujie Tong, Qingxiu Dong, Damai Dai, Yifan song, Tianyu Liu, Baobao Chang, Zhifang Sui

**Abstracts**: Fine-tuning pretrained language models (PLMs) on downstream tasks has become common practice in natural language processing. However, most of the PLMs are vulnerable, e.g., they are brittle under adversarial attacks or imbalanced data, which hinders the application of the PLMs on some downstream tasks, especially in safe-critical scenarios. In this paper, we propose a simple yet effective fine-tuning method called Match-Tuning to force the PLMs to be more robust. For each instance in a batch, we involve other instances in the same batch to interact with it. To be specific, regarding the instances with other labels as a perturbation, Match-Tuning makes the model more robust to noise at the beginning of training. While nearing the end, Match-Tuning focuses more on performing an interpolation among the instances with the same label for better generalization. Extensive experiments on various tasks in GLUE benchmark show that Match-Tuning consistently outperforms the vanilla fine-tuning by $1.64$ scores. Moreover, Match-Tuning exhibits remarkable robustness to adversarial attacks and data imbalance.



## **43. A Word is Worth A Thousand Dollars: Adversarial Attack on Tweets Fools Stock Prediction**

cs.CR

NAACL short paper, github: https://github.com/yonxie/AdvFinTweet

**SubmitDate**: 2022-05-01    [paper-pdf](http://arxiv.org/pdf/2205.01094v1)

**Authors**: Yong Xie, Dakuo Wang, Pin-Yu Chen, Jinjun Xiong, Sijia Liu, Sanmi Koyejo

**Abstracts**: More and more investors and machine learning models rely on social media (e.g., Twitter and Reddit) to gather real-time information and sentiment to predict stock price movements. Although text-based models are known to be vulnerable to adversarial attacks, whether stock prediction models have similar vulnerability is underexplored. In this paper, we experiment with a variety of adversarial attack configurations to fool three stock prediction victim models. We address the task of adversarial generation by solving combinatorial optimization problems with semantics and budget constraints. Our results show that the proposed attack method can achieve consistent success rates and cause significant monetary loss in trading simulation by simply concatenating a perturbed but semantically similar tweet.



## **44. Analysis of a blockchain protocol based on LDPC codes**

cs.CR

**SubmitDate**: 2022-04-30    [paper-pdf](http://arxiv.org/pdf/2202.07265v3)

**Authors**: Massimo Battaglioni, Paolo Santini, Giulia Rafaiani, Franco Chiaraluce, Marco Baldi

**Abstracts**: In a blockchain Data Availability Attack (DAA), a malicious node publishes a block header but withholds part of the block, which contains invalid transactions. Honest full nodes, which can download and store the full blockchain, are aware that some data are not available but they have no formal way to prove it to light nodes, i.e., nodes that have limited resources and are not able to access the whole blockchain data. A common solution to counter these attacks exploits linear error correcting codes to encode the block content. A recent protocol, called SPAR, employs coded Merkle trees and low-density parity-check codes to counter DAAs. In this paper, we show that the protocol is less secure than claimed, owing to a redefinition of the adversarial success probability. As a consequence we show that, for some realistic choices of the parameters, the total amount of data downloaded by light nodes is larger than that obtainable with competitor solutions.



## **45. Optimizing One-pixel Black-box Adversarial Attacks**

cs.CR

9 pasges, 4 figures

**SubmitDate**: 2022-04-30    [paper-pdf](http://arxiv.org/pdf/2205.02116v1)

**Authors**: Tianxun Zhou, Shubhankar Agrawal, Prateek Manocha

**Abstracts**: The output of Deep Neural Networks (DNN) can be altered by a small perturbation of the input in a black box setting by making multiple calls to the DNN. However, the high computation and time required makes the existing approaches unusable. This work seeks to improve the One-pixel (few-pixel) black-box adversarial attacks to reduce the number of calls to the network under attack. The One-pixel attack uses a non-gradient optimization algorithm to find pixel-level perturbations under the constraint of a fixed number of pixels, which causes the network to predict the wrong label for a given image. We show through experimental results how the choice of the optimization algorithm and initial positions to search can reduce function calls and increase attack success significantly, making the attack more practical in real-world settings.



## **46. Logically Consistent Adversarial Attacks for Soft Theorem Provers**

cs.LG

IJCAI-ECAI 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.00047v1)

**Authors**: Alexander Gaskell, Yishu Miao, Lucia Specia, Francesca Toni

**Abstracts**: Recent efforts within the AI community have yielded impressive results towards "soft theorem proving" over natural language sentences using language models. We propose a novel, generative adversarial framework for probing and improving these models' reasoning capabilities. Adversarial attacks in this domain suffer from the logical inconsistency problem, whereby perturbations to the input may alter the label. Our Logically consistent AdVersarial Attacker, LAVA, addresses this by combining a structured generative process with a symbolic solver, guaranteeing logical consistency. Our framework successfully generates adversarial attacks and identifies global weaknesses common across multiple target models. Our analyses reveal naive heuristics and vulnerabilities in these models' reasoning capabilities, exposing an incomplete grasp of logical deduction under logic programs. Finally, in addition to effective probing of these models, we show that training on the generated samples improves the target model's performance.



## **47. To Trust or Not To Trust Prediction Scores for Membership Inference Attacks**

cs.LG

15 pages, 8 figures, 10 tables

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2111.09076v2)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Kristian Kersting

**Abstracts**: Membership inference attacks (MIAs) aim to determine whether a specific sample was used to train a predictive model. Knowing this may indeed lead to a privacy breach. Most MIAs, however, make use of the model's prediction scores - the probability of each output given some input - following the intuition that the trained model tends to behave differently on its training data. We argue that this is a fallacy for many modern deep network architectures. Consequently, MIAs will miserably fail since overconfidence leads to high false-positive rates not only on known domains but also on out-of-distribution data and implicitly acts as a defense against MIAs. Specifically, using generative adversarial networks, we are able to produce a potentially infinite number of samples falsely classified as part of the training data. In other words, the threat of MIAs is overestimated, and less information is leaked than previously assumed. Moreover, there is actually a trade-off between the overconfidence of models and their susceptibility to MIAs: the more classifiers know when they do not know, making low confidence predictions, the more they reveal the training data.



## **48. Adversarial attacks on an optical neural network**

cs.CR

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2205.01226v1)

**Authors**: Shuming Jiao, Ziwei Song, Shuiying Xiang

**Abstracts**: Adversarial attacks have been extensively investigated for machine learning systems including deep learning in the digital domain. However, the adversarial attacks on optical neural networks (ONN) have been seldom considered previously. In this work, we first construct an accurate image classifier with an ONN using a mesh of interconnected Mach-Zehnder interferometers (MZI). Then a corresponding adversarial attack scheme is proposed for the first time. The attacked images are visually very similar to the original ones but the ONN system becomes malfunctioned and generates wrong classification results in most time. The results indicate that adversarial attack is also a significant issue for optical machine learning systems.



## **49. Finding MNEMON: Reviving Memories of Node Embeddings**

cs.LG

To Appear in the 29th ACM Conference on Computer and Communications  Security (CCS), November 7-11, 2022

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.06963v2)

**Authors**: Yun Shen, Yufei Han, Zhikun Zhang, Min Chen, Ting Yu, Michael Backes, Yang Zhang, Gianluca Stringhini

**Abstracts**: Previous security research efforts orbiting around graphs have been exclusively focusing on either (de-)anonymizing the graphs or understanding the security and privacy issues of graph neural networks. Little attention has been paid to understand the privacy risks of integrating the output from graph embedding models (e.g., node embeddings) with complex downstream machine learning pipelines. In this paper, we fill this gap and propose a novel model-agnostic graph recovery attack that exploits the implicit graph structural information preserved in the embeddings of graph nodes. We show that an adversary can recover edges with decent accuracy by only gaining access to the node embedding matrix of the original graph without interactions with the node embedding models. We demonstrate the effectiveness and applicability of our graph recovery attack through extensive experiments.



## **50. Exploration and Exploitation in Federated Learning to Exclude Clients with Poisoned Data**

cs.DC

Accepted at 2022 IWCMC

**SubmitDate**: 2022-04-29    [paper-pdf](http://arxiv.org/pdf/2204.14020v1)

**Authors**: Shadha Tabatabai, Ihab Mohammed, Basheer Qolomany, Abdullatif Albasser, Kashif Ahmad, Mohamed Abdallah, Ala Al-Fuqaha

**Abstracts**: Federated Learning (FL) is one of the hot research topics, and it utilizes Machine Learning (ML) in a distributed manner without directly accessing private data on clients. However, FL faces many challenges, including the difficulty to obtain high accuracy, high communication cost between clients and the server, and security attacks related to adversarial ML. To tackle these three challenges, we propose an FL algorithm inspired by evolutionary techniques. The proposed algorithm groups clients randomly in many clusters, each with a model selected randomly to explore the performance of different models. The clusters are then trained in a repetitive process where the worst performing cluster is removed in each iteration until one cluster remains. In each iteration, some clients are expelled from clusters either due to using poisoned data or low performance. The surviving clients are exploited in the next iteration. The remaining cluster with surviving clients is then used for training the best FL model (i.e., remaining FL model). Communication cost is reduced since fewer clients are used in the final training of the FL model. To evaluate the performance of the proposed algorithm, we conduct a number of experiments using FEMNIST dataset and compare the result against the random FL algorithm. The experimental results show that the proposed algorithm outperforms the baseline algorithm in terms of accuracy, communication cost, and security.



