# Latest Adversarial Attack Papers
**update at 2024-01-09 15:08:02**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Vulnerabilities Unveiled: Adversarially Attacking a Multimodal Vision Language Model for Pathology Imaging**

eess.IV

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.02565v2) [paper-pdf](http://arxiv.org/pdf/2401.02565v2)

**Authors**: Jai Prakash Veerla, Poojitha Thota, Partha Sai Guttikonda, Shirin Nilizadeh, Jacob M. Luber

**Abstract**: In the dynamic landscape of medical artificial intelligence, this study explores the vulnerabilities of the Pathology Language-Image Pretraining (PLIP) model, a Vision Language Foundation model, under targeted adversarial conditions. Leveraging the Kather Colon dataset with 7,180 H&E images across nine tissue types, our investigation employs Projected Gradient Descent (PGD) adversarial attacks to intentionally induce misclassifications. The outcomes reveal a 100% success rate in manipulating PLIP's predictions, underscoring its susceptibility to adversarial perturbations. The qualitative analysis of adversarial examples delves into the interpretability challenges, shedding light on nuanced changes in predictions induced by adversarial manipulations. These findings contribute crucial insights into the interpretability, domain adaptation, and trustworthiness of Vision Language Models in medical imaging. The study emphasizes the pressing need for robust defenses to ensure the reliability of AI models.



## **2. The Impact of Adversarial Node Placement in Decentralized Federated Learning Networks**

cs.CR

Submitted to ICC 2024 conference

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2311.07946v2) [paper-pdf](http://arxiv.org/pdf/2311.07946v2)

**Authors**: Adam Piaseczny, Eric Ruzomberka, Rohit Parasnis, Christopher G. Brinton

**Abstract**: As Federated Learning (FL) grows in popularity, new decentralized frameworks are becoming widespread. These frameworks leverage the benefits of decentralized environments to enable fast and energy-efficient inter-device communication. However, this growing popularity also intensifies the need for robust security measures. While existing research has explored various aspects of FL security, the role of adversarial node placement in decentralized networks remains largely unexplored. This paper addresses this gap by analyzing the performance of decentralized FL for various adversarial placement strategies when adversaries can jointly coordinate their placement within a network. We establish two baseline strategies for placing adversarial node: random placement and network centrality-based placement. Building on this foundation, we propose a novel attack algorithm that prioritizes adversarial spread over adversarial centrality by maximizing the average network distance between adversaries. We show that the new attack algorithm significantly impacts key performance metrics such as testing accuracy, outperforming the baseline frameworks by between 9% and 66.5% for the considered setups. Our findings provide valuable insights into the vulnerabilities of decentralized FL systems, setting the stage for future research aimed at developing more secure and robust decentralized FL frameworks.



## **3. Logits Poisoning Attack in Federated Distillation**

cs.LG

13 pages, 3 figures, 5 tables

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.03685v1) [paper-pdf](http://arxiv.org/pdf/2401.03685v1)

**Authors**: Yuhan Tang, Zhiyuan Wu, Bo Gao, Tian Wen, Yuwei Wang, Sheng Sun

**Abstract**: Federated Distillation (FD) is a novel and promising distributed machine learning paradigm, where knowledge distillation is leveraged to facilitate a more efficient and flexible cross-device knowledge transfer in federated learning. By optimizing local models with knowledge distillation, FD circumvents the necessity of uploading large-scale model parameters to the central server, simultaneously preserving the raw data on local clients. Despite the growing popularity of FD, there is a noticeable gap in previous works concerning the exploration of poisoning attacks within this framework. This can lead to a scant understanding of the vulnerabilities to potential adversarial actions. To this end, we introduce FDLA, a poisoning attack method tailored for FD. FDLA manipulates logit communications in FD, aiming to significantly degrade model performance on clients through misleading the discrimination of private samples. Through extensive simulation experiments across a variety of datasets, attack scenarios, and FD configurations, we demonstrate that LPA effectively compromises client model accuracy, outperforming established baseline algorithms in this regard. Our findings underscore the critical need for robust defense mechanisms in FD settings to mitigate such adversarial threats.



## **4. Assessing the Influence of Different Types of Probing on Adversarial Decision-Making in a Deception Game**

cs.CR

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2310.10662v3) [paper-pdf](http://arxiv.org/pdf/2310.10662v3)

**Authors**: Md Abu Sayed, Mohammad Ariful Islam Khan, Bryant A Allsup, Joshua Zamora, Palvi Aggarwal

**Abstract**: Deception, which includes leading cyber-attackers astray with false information, has shown to be an effective method of thwarting cyber-attacks. There has been little investigation of the effect of probing action costs on adversarial decision-making, despite earlier studies on deception in cybersecurity focusing primarily on variables like network size and the percentage of honeypots utilized in games. Understanding human decision-making when prompted with choices of various costs is essential in many areas such as in cyber security. In this paper, we will use a deception game (DG) to examine different costs of probing on adversarial decisions. To achieve this we utilized an IBLT model and a delayed feedback mechanism to mimic knowledge of human actions. Our results were taken from an even split of deception and no deception to compare each influence. It was concluded that probing was slightly taken less as the cost of probing increased. The proportion of attacks stayed relatively the same as the cost of probing increased. Although a constant cost led to a slight decrease in attacks. Overall, our results concluded that the different probing costs do not have an impact on the proportion of attacks whereas it had a slightly noticeable impact on the proportion of probing.



## **5. AdvSQLi: Generating Adversarial SQL Injections against Real-world WAF-as-a-service**

cs.CR

**SubmitDate**: 2024-01-08    [abs](http://arxiv.org/abs/2401.02615v2) [paper-pdf](http://arxiv.org/pdf/2401.02615v2)

**Authors**: Zhenqing Qu, Xiang Ling, Ting Wang, Xiang Chen, Shouling Ji, Chunming Wu

**Abstract**: As the first defensive layer that attacks would hit, the web application firewall (WAF) plays an indispensable role in defending against malicious web attacks like SQL injection (SQLi). With the development of cloud computing, WAF-as-a-service, as one kind of Security-as-a-service, has been proposed to facilitate the deployment, configuration, and update of WAFs in the cloud. Despite its tremendous popularity, the security vulnerabilities of WAF-as-a-service are still largely unknown, which is highly concerning given its massive usage. In this paper, we propose a general and extendable attack framework, namely AdvSQLi, in which a minimal series of transformations are performed on the hierarchical tree representation of the original SQLi payload, such that the generated SQLi payloads can not only bypass WAF-as-a-service under black-box settings but also keep the same functionality and maliciousness as the original payload. With AdvSQLi, we make it feasible to inspect and understand the security vulnerabilities of WAFs automatically, helping vendors make products more secure.   To evaluate the attack effectiveness and efficiency of AdvSQLi, we first employ two public datasets to generate adversarial SQLi payloads, leading to a maximum attack success rate of 100% against state-of-the-art ML-based SQLi detectors. Furthermore, to demonstrate the immediate security threats caused by AdvSQLi, we evaluate the attack effectiveness against 7 WAF-as-a-service solutions from mainstream vendors and find all of them are vulnerable to AdvSQLi. For instance, AdvSQLi achieves an attack success rate of over 79% against the F5 WAF. Through in-depth analysis of the evaluation results, we further condense out several general yet severe flaws of these vendors that cannot be easily patched.



## **6. Elevating Defenses: Bridging Adversarial Training and Watermarking for Model Resilience**

cs.LG

Accepted at DAI Workshop, AAAI 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2312.14260v2) [paper-pdf](http://arxiv.org/pdf/2312.14260v2)

**Authors**: Janvi Thakkar, Giulio Zizzo, Sergio Maffeis

**Abstract**: Machine learning models are being used in an increasing number of critical applications; thus, securing their integrity and ownership is critical. Recent studies observed that adversarial training and watermarking have a conflicting interaction. This work introduces a novel framework to integrate adversarial training with watermarking techniques to fortify against evasion attacks and provide confident model verification in case of intellectual property theft. We use adversarial training together with adversarial watermarks to train a robust watermarked model. The key intuition is to use a higher perturbation budget to generate adversarial watermarks compared to the budget used for adversarial training, thus avoiding conflict. We use the MNIST and Fashion-MNIST datasets to evaluate our proposed technique on various model stealing attacks. The results obtained consistently outperform the existing baseline in terms of robustness performance and further prove the resilience of this defense against pruning and fine-tuning removal attacks.



## **7. ROIC-DM: Robust Text Inference and Classification via Diffusion Model**

cs.CL

aaai2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.03514v1) [paper-pdf](http://arxiv.org/pdf/2401.03514v1)

**Authors**: Shilong Yuan, Wei Yuan, Tieke HE

**Abstract**: While language models have made many milestones in text inference and classification tasks, they remain susceptible to adversarial attacks that can lead to unforeseen outcomes. Existing works alleviate this problem by equipping language models with defense patches. However, these defense strategies often rely on impractical assumptions or entail substantial sacrifices in model performance. Consequently, enhancing the resilience of the target model using such defense mechanisms is a formidable challenge. This paper introduces an innovative model for robust text inference and classification, built upon diffusion models (ROIC-DM). Benefiting from its training involving denoising stages, ROIC-DM inherently exhibits greater robustness compared to conventional language models. Moreover, ROIC-DM can attain comparable, and in some cases, superior performance to language models, by effectively incorporating them as advisory components. Extensive experiments conducted with several strong textual adversarial attacks on three datasets demonstrate that (1) ROIC-DM outperforms traditional language models in robustness, even when the latter are fortified with advanced defense mechanisms; (2) ROIC-DM can achieve comparable and even better performance than traditional language models by using them as advisors.



## **8. Data-Driven Subsampling in the Presence of an Adversarial Actor**

cs.LG

Accepted for publication at ICMLCN 2024

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2401.03488v1) [paper-pdf](http://arxiv.org/pdf/2401.03488v1)

**Authors**: Abu Shafin Mohammad Mahdee Jameel, Ahmed P. Mohamed, Jinho Yi, Aly El Gamal, Akshay Malhotra

**Abstract**: Deep learning based automatic modulation classification (AMC) has received significant attention owing to its potential applications in both military and civilian use cases. Recently, data-driven subsampling techniques have been utilized to overcome the challenges associated with computational complexity and training time for AMC. Beyond these direct advantages of data-driven subsampling, these methods also have regularizing properties that may improve the adversarial robustness of the modulation classifier. In this paper, we investigate the effects of an adversarial attack on an AMC system that employs deep learning models both for AMC and for subsampling. Our analysis shows that subsampling itself is an effective deterrent to adversarial attacks. We also uncover the most efficient subsampling strategy when an adversarial attack on both the classifier and the subsampler is anticipated.



## **9. Token-Modification Adversarial Attacks for Natural Language Processing: A Survey**

cs.CL

Version 3: edited and expanded

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2103.00676v3) [paper-pdf](http://arxiv.org/pdf/2103.00676v3)

**Authors**: Tom Roth, Yansong Gao, Alsharif Abuadbba, Surya Nepal, Wei Liu

**Abstract**: Many adversarial attacks target natural language processing systems, most of which succeed through modifying the individual tokens of a document. Despite the apparent uniqueness of each of these attacks, fundamentally they are simply a distinct configuration of four components: a goal function, allowable transformations, a search method, and constraints. In this survey, we systematically present the different components used throughout the literature, using an attack-independent framework which allows for easy comparison and categorisation of components. Our work aims to serve as a comprehensive guide for newcomers to the field and to spark targeted research into refining the individual attack components.



## **10. Affinity Uncertainty-based Hard Negative Mining in Graph Contrastive Learning**

cs.LG

Accepted to TNNLS

**SubmitDate**: 2024-01-07    [abs](http://arxiv.org/abs/2301.13340v2) [paper-pdf](http://arxiv.org/pdf/2301.13340v2)

**Authors**: Chaoxi Niu, Guansong Pang, Ling Chen

**Abstract**: Hard negative mining has shown effective in enhancing self-supervised contrastive learning (CL) on diverse data types, including graph CL (GCL). The existing hardness-aware CL methods typically treat negative instances that are most similar to the anchor instance as hard negatives, which helps improve the CL performance, especially on image data. However, this approach often fails to identify the hard negatives but leads to many false negatives on graph data. This is mainly due to that the learned graph representations are not sufficiently discriminative due to oversmooth representations and/or non-independent and identically distributed (non-i.i.d.) issues in graph data. To tackle this problem, this article proposes a novel approach that builds a discriminative model on collective affinity information (i.e., two sets of pairwise affinities between the negative instances and the anchor instance) to mine hard negatives in GCL. In particular, the proposed approach evaluates how confident/uncertain the discriminative model is about the affinity of each negative instance to an anchor instance to determine its hardness weight relative to the anchor instance. This uncertainty information is then incorporated into the existing GCL loss functions via a weighting term to enhance their performance. The enhanced GCL is theoretically grounded that the resulting GCL loss is equivalent to a triplet loss with an adaptive margin being exponentially proportional to the learned uncertainty of each negative instance. Extensive experiments on ten graph datasets show that our approach does the following: 1) consistently enhances different state-of-the-art (SOTA) GCL methods in both graph and node classification tasks and 2) significantly improves their robustness against adversarial attacks. Code is available at https://github.com/mala-lab/AUGCL.



## **11. Data-Dependent Stability Analysis of Adversarial Training**

cs.LG

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03156v1) [paper-pdf](http://arxiv.org/pdf/2401.03156v1)

**Authors**: Yihan Wang, Shuang Liu, Xiao-Shan Gao

**Abstract**: Stability analysis is an essential aspect of studying the generalization ability of deep learning, as it involves deriving generalization bounds for stochastic gradient descent-based training algorithms. Adversarial training is the most widely used defense against adversarial example attacks. However, previous generalization bounds for adversarial training have not included information regarding the data distribution. In this paper, we fill this gap by providing generalization bounds for stochastic gradient descent-based adversarial training that incorporate data distribution information. We utilize the concepts of on-average stability and high-order approximate Lipschitz conditions to examine how changes in data distribution and adversarial budget can affect robust generalization gaps. Our derived generalization bounds for both convex and non-convex losses are at least as good as the uniform stability-based counterparts which do not include data distribution information. Furthermore, our findings demonstrate how distribution shifts from data poisoning attacks can impact robust generalization.



## **12. Transferable Learned Image Compression-Resistant Adversarial Perturbations**

cs.CV

Accepted as poster at Data Compression Conference 2024 (DCC 2024)

**SubmitDate**: 2024-01-06    [abs](http://arxiv.org/abs/2401.03115v1) [paper-pdf](http://arxiv.org/pdf/2401.03115v1)

**Authors**: Yang Sui, Zhuohang Li, Ding Ding, Xiang Pan, Xiaozhong Xu, Shan Liu, Zhenzhong Chen

**Abstract**: Adversarial attacks can readily disrupt the image classification system, revealing the vulnerability of DNN-based recognition tasks. While existing adversarial perturbations are primarily applied to uncompressed images or compressed images by the traditional image compression method, i.e., JPEG, limited studies have investigated the robustness of models for image classification in the context of DNN-based image compression. With the rapid evolution of advanced image compression, DNN-based learned image compression has emerged as the promising approach for transmitting images in many security-critical applications, such as cloud-based face recognition and autonomous driving, due to its superior performance over traditional compression. Therefore, there is a pressing need to fully investigate the robustness of a classification system post-processed by learned image compression. To bridge this research gap, we explore the adversarial attack on a new pipeline that targets image classification models that utilize learned image compressors as pre-processing modules. Furthermore, to enhance the transferability of perturbations across various quality levels and architectures of learned image compression models, we introduce a saliency score-based sampling method to enable the fast generation of transferable perturbation. Extensive experiments with popular attack methods demonstrate the enhanced transferability of our proposed method when attacking images that have been post-processed with different learned image compression models.



## **13. Lotto: Secure Participant Selection against Adversarial Servers in Federated Learning**

cs.CR

20 pages, 14 figures

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02880v1) [paper-pdf](http://arxiv.org/pdf/2401.02880v1)

**Authors**: Zhifeng Jiang, Peng Ye, Shiqi He, Wei Wang, Ruichuan Chen, Bo Li

**Abstract**: In Federated Learning (FL), common privacy-preserving technologies, such as secure aggregation and distributed differential privacy, rely on the critical assumption of an honest majority among participants to withstand various attacks. In practice, however, servers are not always trusted, and an adversarial server can strategically select compromised clients to create a dishonest majority, thereby undermining the system's security guarantees. In this paper, we present Lotto, an FL system that addresses this fundamental, yet underexplored issue by providing secure participant selection against an adversarial server. Lotto supports two selection algorithms: random and informed. To ensure random selection without a trusted server, Lotto enables each client to autonomously determine their participation using verifiable randomness. For informed selection, which is more vulnerable to manipulation, Lotto approximates the algorithm by employing random selection within a refined client pool. Our theoretical analysis shows that Lotto effectively restricts the number of server-selected compromised clients, thus ensuring an honest majority among participants. Large-scale experiments further reveal that Lotto achieves time-to-accuracy performance comparable to that of insecure selection methods, indicating a low computational overhead for secure selection.



## **14. PromptBench: A Unified Library for Evaluation of Large Language Models**

cs.AI

An extension to PromptBench (arXiv:2306.04528) for unified evaluation  of LLMs using the same name; code: https://github.com/microsoft/promptbench

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2312.07910v2) [paper-pdf](http://arxiv.org/pdf/2312.07910v2)

**Authors**: Kaijie Zhu, Qinlin Zhao, Hao Chen, Jindong Wang, Xing Xie

**Abstract**: The evaluation of large language models (LLMs) is crucial to assess their performance and mitigate potential security risks. In this paper, we introduce PromptBench, a unified library to evaluate LLMs. It consists of several key components that are easily used and extended by researchers: prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools. PromptBench is designed to be an open, general, and flexible codebase for research purposes that can facilitate original study in creating new benchmarks, deploying downstream applications, and designing new evaluation protocols. The code is available at: https://github.com/microsoft/promptbench and will be continuously supported.



## **15. Enhancing targeted transferability via feature space fine-tuning**

cs.CV

9 pages, 10 figures, accepted by 2024ICASSP

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02727v1) [paper-pdf](http://arxiv.org/pdf/2401.02727v1)

**Authors**: Hui Zeng, Biwei Chen, Anjie Peng

**Abstract**: Adversarial examples (AEs) have been extensively studied due to their potential for privacy protection and inspiring robust neural networks. However, making a targeted AE transferable across unknown models remains challenging. In this paper, to alleviate the overfitting dilemma common in an AE crafted by existing simple iterative attacks, we propose fine-tuning it in the feature space. Specifically, starting with an AE generated by a baseline attack, we encourage the features that contribute to the target class and discourage the features that contribute to the original class in a middle layer of the source model. Extensive experiments demonstrate that only a few iterations of fine-tuning can boost existing attacks in terms of targeted transferability nontrivially and universally. Our results also verify that the simple iterative attacks can yield comparable or even better transferability than the resource-intensive methods, which rely on training target-specific classifiers or generators with additional data. The code is available at: github.com/zengh5/TA_feature_FT.



## **16. Calibration Attack: A Framework For Adversarial Attacks Targeting Calibration**

cs.LG

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02718v1) [paper-pdf](http://arxiv.org/pdf/2401.02718v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu, Hongyu Guo

**Abstract**: We introduce a new framework of adversarial attacks, named calibration attacks, in which the attacks are generated and organized to trap victim models to be miscalibrated without altering their original accuracy, hence seriously endangering the trustworthiness of the models and any decision-making based on their confidence scores. Specifically, we identify four novel forms of calibration attacks: underconfidence attacks, overconfidence attacks, maximum miscalibration attacks, and random confidence attacks, in both the black-box and white-box setups. We then test these new attacks on typical victim models with comprehensive datasets, demonstrating that even with a relatively low number of queries, the attacks can create significant calibration mistakes. We further provide detailed analyses to understand different aspects of calibration attacks. Building on that, we investigate the effectiveness of widely used adversarial defences and calibration methods against these types of attacks, which then inspires us to devise two novel defences against such calibration attacks.



## **17. Game Theory for Adversarial Attacks and Defenses**

cs.LG

With the agreement of my coauthors, I would like to withdraw the  manuscript "Game Theory for Adversarial Attacks and Defenses". Some  experimental procedures were not included in the manuscript, which makes a  part of important claims not meaningful

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2110.06166v4) [paper-pdf](http://arxiv.org/pdf/2110.06166v4)

**Authors**: Shorya Sharma

**Abstract**: Adversarial attacks can generate adversarial inputs by applying small but intentionally worst-case perturbations to samples from the dataset, which leads to even state-of-the-art deep neural networks outputting incorrect answers with high confidence. Hence, some adversarial defense techniques are developed to improve the security and robustness of the models and avoid them being attacked. Gradually, a game-like competition between attackers and defenders formed, in which both players would attempt to play their best strategies against each other while maximizing their own payoffs. To solve the game, each player would choose an optimal strategy against the opponent based on the prediction of the opponent's strategy choice. In this work, we are on the defensive side to apply game-theoretic approaches on defending against attacks. We use two randomization methods, random initialization and stochastic activation pruning, to create diversity of networks. Furthermore, we use one denoising technique, super resolution, to improve models' robustness by preprocessing images before attacks. Our experimental results indicate that those three methods can effectively improve the robustness of deep-learning neural networks.



## **18. Secure Control of Connected and Automated Vehicles Using Trust-Aware Robust Event-Triggered Control Barrier Functions**

eess.SY

arXiv admin note: substantial text overlap with arXiv:2305.16818

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02306v2) [paper-pdf](http://arxiv.org/pdf/2401.02306v2)

**Authors**: H M Sabbir Ahmad, Ehsan Sabouni, Akua Dickson, Wei Xiao, Christos G. Cassandras, Wenchao Li

**Abstract**: We address the security of a network of Connected and Automated Vehicles (CAVs) cooperating to safely navigate through a conflict area (e.g., traffic intersections, merging roadways, roundabouts). Previous studies have shown that such a network can be targeted by adversarial attacks causing traffic jams or safety violations ending in collisions. We focus on attacks targeting the V2X communication network used to share vehicle data and consider as well uncertainties due to noise in sensor measurements and communication channels. To combat these, motivated by recent work on the safe control of CAVs, we propose a trust-aware robust event-triggered decentralized control and coordination framework that can provably guarantee safety. We maintain a trust metric for each vehicle in the network computed based on their behavior and used to balance the tradeoff between conservativeness (when deeming every vehicle as untrustworthy) and guaranteed safety and security. It is important to highlight that our framework is invariant to the specific choice of the trust framework. Based on this framework, we propose an attack detection and mitigation scheme which has twofold benefits: (i) the trust framework is immune to false positives, and (ii) it provably guarantees safety against false positive cases. We use extensive simulations (in SUMO and CARLA) to validate the theoretical guarantees and demonstrate the efficacy of our proposed scheme to detect and mitigate adversarial attacks.



## **19. MalModel: Hiding Malicious Payload in Mobile Deep Learning Models with Black-box Backdoor Attack**

cs.CR

Due to the limitation "The abstract field cannot be longer than 1,920  characters", the abstract here is shorter than that in the PDF file

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02659v1) [paper-pdf](http://arxiv.org/pdf/2401.02659v1)

**Authors**: Jiayi Hua, Kailong Wang, Meizhen Wang, Guangdong Bai, Xiapu Luo, Haoyu Wang

**Abstract**: Mobile malware has become one of the most critical security threats in the era of ubiquitous mobile computing. Despite the intensive efforts from security experts to counteract it, recent years have still witnessed a rapid growth of identified malware samples. This could be partly attributed to the newly-emerged technologies that may constantly open up under-studied attack surfaces for the adversaries. One typical example is the recently-developed mobile machine learning (ML) framework that enables storing and running deep learning (DL) models on mobile devices. Despite obvious advantages, this new feature also inadvertently introduces potential vulnerabilities (e.g., on-device models may be modified for malicious purposes). In this work, we propose a method to generate or transform mobile malware by hiding the malicious payloads inside the parameters of deep learning models, based on a strategy that considers four factors (layer type, layer number, layer coverage and the number of bytes to replace). Utilizing the proposed method, we can run malware in DL mobile applications covertly with little impact on the model performance (i.e., as little as 0.4% drop in accuracy and at most 39ms latency overhead).



## **20. A Random Ensemble of Encrypted models for Enhancing Robustness against Adversarial Examples**

cs.CR

4 pages

**SubmitDate**: 2024-01-05    [abs](http://arxiv.org/abs/2401.02633v1) [paper-pdf](http://arxiv.org/pdf/2401.02633v1)

**Authors**: Ryota Iijima, Sayaka Shiota, Hitoshi Kiya

**Abstract**: Deep neural networks (DNNs) are well known to be vulnerable to adversarial examples (AEs). In addition, AEs have adversarial transferability, which means AEs generated for a source model can fool another black-box model (target model) with a non-trivial probability. In previous studies, it was confirmed that the vision transformer (ViT) is more robust against the property of adversarial transferability than convolutional neural network (CNN) models such as ConvMixer, and moreover encrypted ViT is more robust than ViT without any encryption. In this article, we propose a random ensemble of encrypted ViT models to achieve much more robust models. In experiments, the proposed scheme is verified to be more robust against not only black-box attacks but also white-box ones than convention methods.



## **21. A Practical Survey on Emerging Threats from AI-driven Voice Attacks: How Vulnerable are Commercial Voice Control Systems?**

cs.CR

14 pages

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.06010v2) [paper-pdf](http://arxiv.org/pdf/2312.06010v2)

**Authors**: Yuanda Wang, Qiben Yan, Nikolay Ivanov, Xun Chen

**Abstract**: The emergence of Artificial Intelligence (AI)-driven audio attacks has revealed new security vulnerabilities in voice control systems. While researchers have introduced a multitude of attack strategies targeting voice control systems (VCS), the continual advancements of VCS have diminished the impact of many such attacks. Recognizing this dynamic landscape, our study endeavors to comprehensively assess the resilience of commercial voice control systems against a spectrum of malicious audio attacks. Through extensive experimentation, we evaluate six prominent attack techniques across a collection of voice control interfaces and devices. Contrary to prevailing narratives, our results suggest that commercial voice control systems exhibit enhanced resistance to existing threats. Particularly, our research highlights the ineffectiveness of white-box attacks in black-box scenarios. Furthermore, the adversaries encounter substantial obstacles in obtaining precise gradient estimations during query-based interactions with commercial systems, such as Apple Siri and Samsung Bixby. Meanwhile, we find that current defense strategies are not completely immune to advanced attacks. Our findings contribute valuable insights for enhancing defense mechanisms in VCS. Through this survey, we aim to raise awareness within the academic community about the security concerns of VCS and advocate for continued research in this crucial area.



## **22. Evasive Hardware Trojan through Adversarial Power Trace**

cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2401.02342v1) [paper-pdf](http://arxiv.org/pdf/2401.02342v1)

**Authors**: Behnam Omidi, Khaled N. Khasawneh, Ihsen Alouani

**Abstract**: The globalization of the Integrated Circuit (IC) supply chain, driven by time-to-market and cost considerations, has made ICs vulnerable to hardware Trojans (HTs). Against this threat, a promising approach is to use Machine Learning (ML)-based side-channel analysis, which has the advantage of being a non-intrusive method, along with efficiently detecting HTs under golden chip-free settings. In this paper, we question the trustworthiness of ML-based HT detection via side-channel analysis. We introduce a HT obfuscation (HTO) approach to allow HTs to bypass this detection method. Rather than theoretically misleading the model by simulated adversarial traces, a key aspect of our approach is the design and implementation of adversarial noise as part of the circuitry, alongside the HT. We detail HTO methodologies for ASICs and FPGAs, and evaluate our approach using TrustHub benchmark. Interestingly, we found that HTO can be implemented with only a single transistor for ASIC designs to generate adversarial power traces that can fool the defense with 100% efficiency. We also efficiently implemented our approach on a Spartan 6 Xilinx FPGA using 2 different variants: (i) DSP slices-based, and (ii) ring-oscillator-based design. Additionally, we assess the efficiency of countermeasures like spectral domain analysis, and we show that an adaptive attacker can still design evasive HTOs by constraining the design with a spectral noise budget. In addition, while adversarial training (AT) offers higher protection against evasive HTs, AT models suffer from a considerable utility loss, potentially rendering them unsuitable for such security application. We believe this research represents a significant step in understanding and exploiting ML vulnerabilities in a hardware security context, and we make all resources and designs openly available online: https://dev.d18uu4lqwhbmka.amplifyapp.com



## **23. Adversarial Data Poisoning for Fake News Detection: How to Make a Model Misclassify a Target News without Modifying It**

cs.LG

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.15228v2) [paper-pdf](http://arxiv.org/pdf/2312.15228v2)

**Authors**: Federico Siciliano, Luca Maiano, Lorenzo Papa, Federica Baccini, Irene Amerini, Fabrizio Silvestri

**Abstract**: Fake news detection models are critical to countering disinformation but can be manipulated through adversarial attacks. In this position paper, we analyze how an attacker can compromise the performance of an online learning detector on specific news content without being able to manipulate the original target news. In some contexts, such as social networks, where the attacker cannot exert complete control over all the information, this scenario can indeed be quite plausible. Therefore, we show how an attacker could potentially introduce poisoning data into the training data to manipulate the behavior of an online learning method. Our initial findings reveal varying susceptibility of logistic regression models based on complexity and attack type.



## **24. Attacks in Adversarial Machine Learning: A Systematic Survey from the Life-cycle Perspective**

cs.LG

35 pages, 4 figures, 10 tables, 313 reference papers

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2302.09457v2) [paper-pdf](http://arxiv.org/pdf/2302.09457v2)

**Authors**: Baoyuan Wu, Zihao Zhu, Li Liu, Qingshan Liu, Zhaofeng He, Siwei Lyu

**Abstract**: Adversarial machine learning (AML) studies the adversarial phenomenon of machine learning, which may make inconsistent or unexpected predictions with humans. Some paradigms have been recently developed to explore this adversarial phenomenon occurring at different stages of a machine learning system, such as backdoor attack occurring at the pre-training, in-training and inference stage; weight attack occurring at the post-training, deployment and inference stage; adversarial attack occurring at the inference stage. However, although these adversarial paradigms share a common goal, their developments are almost independent, and there is still no big picture of AML. In this work, we aim to provide a unified perspective to the AML community to systematically review the overall progress of this field. We firstly provide a general definition about AML, and then propose a unified mathematical framework to covering existing attack paradigms. According to the proposed unified framework, we build a full taxonomy to systematically categorize and review existing representative methods for each paradigm. Besides, using this unified framework, it is easy to figure out the connections and differences among different attack paradigms, which may inspire future researchers to develop more advanced attack paradigms. Finally, to facilitate the viewing of the built taxonomy and the related literature in adversarial machine learning, we further provide a website, \ie, \url{http://adversarial-ml.com}, where the taxonomies and literature will be continuously updated.



## **25. Fast Certification of Vision-Language Models Using Incremental Randomized Smoothing**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.09024v2) [paper-pdf](http://arxiv.org/pdf/2311.09024v2)

**Authors**: A K Nirala, A Joshi, C Hegde, S Sarkar

**Abstract**: A key benefit of deep vision-language models such as CLIP is that they enable zero-shot open vocabulary classification; the user has the ability to define novel class labels via natural language prompts at inference time. However, while CLIP-based zero-shot classifiers have demonstrated competitive performance across a range of domain shifts, they remain highly vulnerable to adversarial attacks. Therefore, ensuring the robustness of such models is crucial for their reliable deployment in the wild.   In this work, we introduce Open Vocabulary Certification (OVC), a fast certification method designed for open-vocabulary models like CLIP via randomized smoothing techniques. Given a base "training" set of prompts and their corresponding certified CLIP classifiers, OVC relies on the observation that a classifier with a novel prompt can be viewed as a perturbed version of nearby classifiers in the base training set. Therefore, OVC can rapidly certify the novel classifier using a variation of incremental randomized smoothing. By using a caching trick, we achieve approximately two orders of magnitude acceleration in the certification process for novel prompts. To achieve further (heuristic) speedups, OVC approximates the embedding space at a given input using a multivariate normal distribution bypassing the need for sampling via forward passes through the vision backbone. We demonstrate the effectiveness of OVC on through experimental evaluation using multiple vision-language backbones on the CIFAR-10 and ImageNet test datasets.



## **26. InstructTA: Instruction-Tuned Targeted Attack for Large Vision-Language Models**

cs.CV

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2312.01886v2) [paper-pdf](http://arxiv.org/pdf/2312.01886v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Pingchuan Ma, Zongjie Li, Shuai Wang

**Abstract**: Large vision-language models (LVLMs) have demonstrated their incredible capability in image understanding and response generation. However, this rich visual interaction also makes LVLMs vulnerable to adversarial examples. In this paper, we formulate a novel and practical gray-box attack scenario that the adversary can only access the visual encoder of the victim LVLM, without the knowledge of its prompts (which are often proprietary for service providers and not publicly available) and its underlying large language model (LLM). This practical setting poses challenges to the cross-prompt and cross-model transferability of targeted adversarial attack, which aims to confuse the LVLM to output a response that is semantically similar to the attacker's chosen target text. To this end, we propose an instruction-tuned targeted attack (dubbed InstructTA) to deliver the targeted adversarial attack on LVLMs with high transferability. Initially, we utilize a public text-to-image generative model to "reverse" the target response into a target image, and employ GPT-4 to infer a reasonable instruction $\boldsymbol{p}^\prime$ from the target response. We then form a local surrogate model (sharing the same visual encoder with the victim LVLM) to extract instruction-aware features of an adversarial image example and the target image, and minimize the distance between these two features to optimize the adversarial example. To further improve the transferability, we augment the instruction $\boldsymbol{p}^\prime$ with instructions paraphrased from an LLM. Extensive experiments demonstrate the superiority of our proposed method in targeted attack performance and transferability.



## **27. DiffAttack: Evasion Attacks Against Diffusion-Based Adversarial Purification**

cs.CR

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2311.16124v2) [paper-pdf](http://arxiv.org/pdf/2311.16124v2)

**Authors**: Mintong Kang, Dawn Song, Bo Li

**Abstract**: Diffusion-based purification defenses leverage diffusion models to remove crafted perturbations of adversarial examples and achieve state-of-the-art robustness. Recent studies show that even advanced attacks cannot break such defenses effectively, since the purification process induces an extremely deep computational graph which poses the potential problem of gradient obfuscation, high memory cost, and unbounded randomness. In this paper, we propose a unified framework DiffAttack to perform effective and efficient attacks against diffusion-based purification defenses, including both DDPM and score-based approaches. In particular, we propose a deviated-reconstruction loss at intermediate diffusion steps to induce inaccurate density gradient estimation to tackle the problem of vanishing/exploding gradients. We also provide a segment-wise forwarding-backwarding algorithm, which leads to memory-efficient gradient backpropagation. We validate the attack effectiveness of DiffAttack compared with existing adaptive attacks on CIFAR-10 and ImageNet. We show that DiffAttack decreases the robust accuracy of models compared with SOTA attacks by over 20% on CIFAR-10 under $\ell_\infty$ attack $(\epsilon=8/255)$, and over 10% on ImageNet under $\ell_\infty$ attack $(\epsilon=4/255)$. We conduct a series of ablations studies, and we find 1) DiffAttack with the deviated-reconstruction loss added over uniformly sampled time steps is more effective than that added over only initial/final steps, and 2) diffusion-based purification with a moderate diffusion length is more robust under DiffAttack.



## **28. CBD: A Certified Backdoor Detector Based on Local Dominant Probability**

cs.LG

Accepted to NeurIPS 2023

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2310.17498v2) [paper-pdf](http://arxiv.org/pdf/2310.17498v2)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor attack is a common threat to deep neural networks. During testing, samples embedded with a backdoor trigger will be misclassified as an adversarial target by a backdoored model, while samples without the backdoor trigger will be correctly classified. In this paper, we present the first certified backdoor detector (CBD), which is based on a novel, adjustable conformal prediction scheme based on our proposed statistic local dominant probability. For any classifier under inspection, CBD provides 1) a detection inference, 2) the condition under which the attacks are guaranteed to be detectable for the same classification domain, and 3) a probabilistic upper bound for the false positive rate. Our theoretical results show that attacks with triggers that are more resilient to test-time noise and have smaller perturbation magnitudes are more likely to be detected with guarantees. Moreover, we conduct extensive experiments on four benchmark datasets considering various backdoor types, such as BadNet, CB, and Blend. CBD achieves comparable or even higher detection accuracy than state-of-the-art detectors, and it in addition provides detection certification. Notably, for backdoor attacks with random perturbation triggers bounded by $\ell_2\leq0.75$ which achieves more than 90\% attack success rate, CBD achieves 100\% (98\%), 100\% (84\%), 98\% (98\%), and 72\% (40\%) empirical (certified) detection true positive rates on the four benchmark datasets GTSRB, SVHN, CIFAR-10, and TinyImageNet, respectively, with low false positive rates.



## **29. DeepTaster: Adversarial Perturbation-Based Fingerprinting to Identify Proprietary Dataset Use in Deep Neural Networks**

cs.CR

**SubmitDate**: 2024-01-04    [abs](http://arxiv.org/abs/2211.13535v2) [paper-pdf](http://arxiv.org/pdf/2211.13535v2)

**Authors**: Seonhye Park, Alsharif Abuadbba, Shuo Wang, Kristen Moore, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: Training deep neural networks (DNNs) requires large datasets and powerful computing resources, which has led some owners to restrict redistribution without permission. Watermarking techniques that embed confidential data into DNNs have been used to protect ownership, but these can degrade model performance and are vulnerable to watermark removal attacks. Recently, DeepJudge was introduced as an alternative approach to measuring the similarity between a suspect and a victim model. While DeepJudge shows promise in addressing the shortcomings of watermarking, it primarily addresses situations where the suspect model copies the victim's architecture. In this study, we introduce DeepTaster, a novel DNN fingerprinting technique, to address scenarios where a victim's data is unlawfully used to build a suspect model. DeepTaster can effectively identify such DNN model theft attacks, even when the suspect model's architecture deviates from the victim's. To accomplish this, DeepTaster generates adversarial images with perturbations, transforms them into the Fourier frequency domain, and uses these transformed images to identify the dataset used in a suspect model. The underlying premise is that adversarial images can capture the unique characteristics of DNNs built with a specific dataset. To demonstrate the effectiveness of DeepTaster, we evaluated the effectiveness of DeepTaster by assessing its detection accuracy on three datasets (CIFAR10, MNIST, and Tiny-ImageNet) across three model architectures (ResNet18, VGG16, and DenseNet161). We conducted experiments under various attack scenarios, including transfer learning, pruning, fine-tuning, and data augmentation. Specifically, in the Multi-Architecture Attack scenario, DeepTaster was able to identify all the stolen cases across all datasets, while DeepJudge failed to detect any of the cases.



## **30. Integrated Cyber-Physical Resiliency for Power Grids under IoT-Enabled Dynamic Botnet Attacks**

eess.SY

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01963v1) [paper-pdf](http://arxiv.org/pdf/2401.01963v1)

**Authors**: Yuhan Zhao, Juntao Chen, Quanyan Zhu

**Abstract**: The wide adoption of Internet of Things (IoT)-enabled energy devices improves the quality of life, but simultaneously, it enlarges the attack surface of the power grid system. The adversary can gain illegitimate control of a large number of these devices and use them as a means to compromise the physical grid operation, a mechanism known as the IoT botnet attack. This paper aims to improve the resiliency of cyber-physical power grids to such attacks. Specifically, we use an epidemic model to understand the dynamic botnet formation, which facilitates the assessment of the cyber layer vulnerability of the grid. The attacker aims to exploit this vulnerability to enable a successful physical compromise, while the system operator's goal is to ensure a normal operation of the grid by mitigating cyber risks. We develop a cross-layer game-theoretic framework for strategic decision-making to enhance cyber-physical grid resiliency. The cyber-layer game guides the system operator on how to defend against the botnet attacker as the first layer of defense, while the dynamic game strategy at the physical layer further counteracts the adversarial behavior in real time for improved physical resilience. A number of case studies on the IEEE-39 bus system are used to corroborate the devised approach.



## **31. Mining Temporal Attack Patterns from Cyberthreat Intelligence Reports**

cs.CR

A modified version of this pre-print is submitted to IEEE  Transactions on Software Engineering, and is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01883v1) [paper-pdf](http://arxiv.org/pdf/2401.01883v1)

**Authors**: Md Rayhanur Rahman, Brandon Wroblewski, Quinn Matthews, Brantley Morgan, Tim Menzies, Laurie Williams

**Abstract**: Defending from cyberattacks requires practitioners to operate on high-level adversary behavior. Cyberthreat intelligence (CTI) reports on past cyberattack incidents describe the chain of malicious actions with respect to time. To avoid repeating cyberattack incidents, practitioners must proactively identify and defend against recurring chain of actions - which we refer to as temporal attack patterns. Automatically mining the patterns among actions provides structured and actionable information on the adversary behavior of past cyberattacks. The goal of this paper is to aid security practitioners in prioritizing and proactive defense against cyberattacks by mining temporal attack patterns from cyberthreat intelligence reports. To this end, we propose ChronoCTI, an automated pipeline for mining temporal attack patterns from cyberthreat intelligence (CTI) reports of past cyberattacks. To construct ChronoCTI, we build the ground truth dataset of temporal attack patterns and apply state-of-the-art large language models, natural language processing, and machine learning techniques. We apply ChronoCTI on a set of 713 CTI reports, where we identify 124 temporal attack patterns - which we categorize into nine pattern categories. We identify that the most prevalent pattern category is to trick victim users into executing malicious code to initiate the attack, followed by bypassing the anti-malware system in the victim network. Based on the observed patterns, we advocate organizations to train users about cybersecurity best practices, introduce immutable operating systems with limited functionalities, and enforce multi-user authentications. Moreover, we advocate practitioners to leverage the automated mining capability of ChronoCTI and design countermeasures against the recurring attack patterns.



## **32. Attackers reveal their arsenal: An investigation of adversarial techniques in CTI reports**

cs.CR

This version is submitted to ACM Transactions on Privacy and  Security. This version is under review

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01865v1) [paper-pdf](http://arxiv.org/pdf/2401.01865v1)

**Authors**: Md Rayhanur Rahman, Setu Kumar Basak, Rezvan Mahdavi Hezaveh, Laurie Williams

**Abstract**: Context: Cybersecurity vendors often publish cyber threat intelligence (CTI) reports, referring to the written artifacts on technical and forensic analysis of the techniques used by the malware in APT attacks. Objective: The goal of this research is to inform cybersecurity practitioners about how adversaries form cyberattacks through an analysis of adversarial techniques documented in cyberthreat intelligence reports. Dataset: We use 594 adversarial techniques cataloged in MITRE ATT\&CK. We systematically construct a set of 667 CTI reports that MITRE ATT\&CK used as citations in the descriptions of the cataloged adversarial techniques. Methodology: We analyze the frequency and trend of adversarial techniques, followed by a qualitative analysis of the implementation of techniques. Next, we perform association rule mining to identify pairs of techniques recurring in APT attacks. We then perform qualitative analysis to identify the underlying relations among the techniques in the recurring pairs. Findings: The set of 667 CTI reports documents 10,370 techniques in total, and we identify 19 prevalent techniques accounting for 37.3\% of documented techniques. We also identify 425 statistically significant recurring pairs and seven types of relations among the techniques in these pairs. The top three among the seven relationships suggest that techniques used by the malware inter-relate with one another in terms of (a) abusing or affecting the same system assets, (b) executing in sequences, and (c) overlapping in their implementations. Overall, the study quantifies how adversaries leverage techniques through malware in APT attacks based on publicly reported documents. We advocate organizations prioritize their defense against the identified prevalent techniques and actively hunt for potential malicious intrusion based on the identified pairs of techniques.



## **33. Locally Differentially Private Embedding Models in Distributed Fraud Prevention Systems**

cs.CR

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.02450v1) [paper-pdf](http://arxiv.org/pdf/2401.02450v1)

**Authors**: Iker Perez, Jason Wong, Piotr Skalski, Stuart Burrell, Richard Mortier, Derek McAuley, David Sutton

**Abstract**: Global financial crime activity is driving demand for machine learning solutions in fraud prevention. However, prevention systems are commonly serviced to financial institutions in isolation, and few provisions exist for data sharing due to fears of unintentional leaks and adversarial attacks. Collaborative learning advances in finance are rare, and it is hard to find real-world insights derived from privacy-preserving data processing systems. In this paper, we present a collaborative deep learning framework for fraud prevention, designed from a privacy standpoint, and awarded at the recent PETs Prize Challenges. We leverage latent embedded representations of varied-length transaction sequences, along with local differential privacy, in order to construct a data release mechanism which can securely inform externally hosted fraud and anomaly detection models. We assess our contribution on two distributed data sets donated by large payment networks, and demonstrate robustness to popular inference-time attacks, along with utility-privacy trade-offs analogous to published work in alternative application domains.



## **34. Towards Robust Semantic Segmentation against Patch-based Attack via Attention Refinement**

cs.CV

30 pages, 3 figures, 12 tables

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01750v1) [paper-pdf](http://arxiv.org/pdf/2401.01750v1)

**Authors**: Zheng Yuan, Jie Zhang, Yude Wang, Shiguang Shan, Xilin Chen

**Abstract**: The attention mechanism has been proven effective on various visual tasks in recent years. In the semantic segmentation task, the attention mechanism is applied in various methods, including the case of both Convolution Neural Networks (CNN) and Vision Transformer (ViT) as backbones. However, we observe that the attention mechanism is vulnerable to patch-based adversarial attacks. Through the analysis of the effective receptive field, we attribute it to the fact that the wide receptive field brought by global attention may lead to the spread of the adversarial patch. To address this issue, in this paper, we propose a Robust Attention Mechanism (RAM) to improve the robustness of the semantic segmentation model, which can notably relieve the vulnerability against patch-based attacks. Compared to the vallina attention mechanism, RAM introduces two novel modules called Max Attention Suppression and Random Attention Dropout, both of which aim to refine the attention matrix and limit the influence of a single adversarial patch on the semantic segmentation results of other positions. Extensive experiments demonstrate the effectiveness of our RAM to improve the robustness of semantic segmentation models against various patch-based attack methods under different attack settings.



## **35. An Initial Investigation of Neural Replay Simulator for Over-the-Air Adversarial Perturbations to Automatic Speaker Verification**

cs.SD

Accepted in ICASSP 2024

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2310.05354v4) [paper-pdf](http://arxiv.org/pdf/2310.05354v4)

**Authors**: Jiaqi Li, Li Wang, Liumeng Xue, Lei Wang, Zhizheng Wu

**Abstract**: Deep Learning has advanced Automatic Speaker Verification (ASV) in the past few years. Although it is known that deep learning-based ASV systems are vulnerable to adversarial examples in digital access, there are few studies on adversarial attacks in the context of physical access, where a replay process (i.e., over the air) is involved. An over-the-air attack involves a loudspeaker, a microphone, and a replaying environment that impacts the movement of the sound wave. Our initial experiment confirms that the replay process impacts the effectiveness of the over-the-air attack performance. This study performs an initial investigation towards utilizing a neural replay simulator to improve over-the-air adversarial attack robustness. This is achieved by using a neural waveform synthesizer to simulate the replay process when estimating the adversarial perturbations. Experiments conducted on the ASVspoof2019 dataset confirm that the neural replay simulator can considerably increase the success rates of over-the-air adversarial attacks. This raises the concern for adversarial attacks on speaker verification in physical access applications.



## **36. Will 6G be Semantic Communications? Opportunities and Challenges from Task Oriented and Secure Communications to Integrated Sensing**

cs.NI

**SubmitDate**: 2024-01-03    [abs](http://arxiv.org/abs/2401.01531v1) [paper-pdf](http://arxiv.org/pdf/2401.01531v1)

**Authors**: Yalin E. Sagduyu, Tugba Erpek, Aylin Yener, Sennur Ulukus

**Abstract**: This paper explores opportunities and challenges of task (goal)-oriented and semantic communications for next-generation (NextG) communication networks through the integration of multi-task learning. This approach employs deep neural networks representing a dedicated encoder at the transmitter and multiple task-specific decoders at the receiver, collectively trained to handle diverse tasks including semantic information preservation, source input reconstruction, and integrated sensing and communications. To extend the applicability from point-to-point links to multi-receiver settings, we envision the deployment of decoders at various receivers, where decentralized learning addresses the challenges of communication load and privacy concerns, leveraging federated learning techniques that distribute model updates across decentralized nodes. However, the efficacy of this approach is contingent on the robustness of the employed deep learning models. We scrutinize potential vulnerabilities stemming from adversarial attacks during both training and testing phases. These attacks aim to manipulate both the inputs at the encoder at the transmitter and the signals received over the air on the receiver side, highlighting the importance of fortifying semantic communications against potential multi-domain exploits. Overall, the joint and robust design of task-oriented communications, semantic communications, and integrated sensing and communications in a multi-task learning framework emerges as the key enabler for context-aware, resource-efficient, and secure communications ultimately needed in NextG network systems.



## **37. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

cs.LG

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01199v1) [paper-pdf](http://arxiv.org/pdf/2401.01199v1)

**Authors**: Benedetta Tondi, Wei Guo, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, we propose a more general, theoretically sound, targeted attack that resorts to the minimization of a Jacobian-induced MAhalanobis distance (JMA) term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm provides an optimal solution to a linearized version of the adversarial example problem originally introduced by Szegedy et al. \cite{szegedy2013intriguing}. The experiments we carried out confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, the JMA attack is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in a complex multilabel classification scenario with 20 labels, a capability that is out of reach of all the attacks proposed so far. As a further advantage, the JMA attack usually requires very few iterations, thus resulting more efficient than existing methods.



## **38. Dual Teacher Knowledge Distillation with Domain Alignment for Face Anti-spoofing**

cs.CV

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01102v1) [paper-pdf](http://arxiv.org/pdf/2401.01102v1)

**Authors**: Zhe Kong, Wentian Zhang, Tao Wang, Kaihao Zhang, Yuexiang Li, Xiaoying Tang, Wenhan Luo

**Abstract**: Face recognition systems have raised concerns due to their vulnerability to different presentation attacks, and system security has become an increasingly critical concern. Although many face anti-spoofing (FAS) methods perform well in intra-dataset scenarios, their generalization remains a challenge. To address this issue, some methods adopt domain adversarial training (DAT) to extract domain-invariant features. However, the competition between the encoder and the domain discriminator can cause the network to be difficult to train and converge. In this paper, we propose a domain adversarial attack (DAA) method to mitigate the training instability problem by adding perturbations to the input images, which makes them indistinguishable across domains and enables domain alignment. Moreover, since models trained on limited data and types of attacks cannot generalize well to unknown attacks, we propose a dual perceptual and generative knowledge distillation framework for face anti-spoofing that utilizes pre-trained face-related models containing rich face priors. Specifically, we adopt two different face-related models as teachers to transfer knowledge to the target student model. The pre-trained teacher models are not from the task of face anti-spoofing but from perceptual and generative tasks, respectively, which implicitly augment the data. By combining both DAA and dual-teacher knowledge distillation, we develop a dual teacher knowledge distillation with domain alignment framework (DTDA) for face anti-spoofing. The advantage of our proposed method has been verified through extensive ablation studies and comparison with state-of-the-art methods on public datasets across multiple protocols.



## **39. Imperio: Language-Guided Backdoor Attacks for Arbitrary Model Control**

cs.CR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.01085v1) [paper-pdf](http://arxiv.org/pdf/2401.01085v1)

**Authors**: Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: Revolutionized by the transformer architecture, natural language processing (NLP) has received unprecedented attention. While advancements in NLP models have led to extensive research into their backdoor vulnerabilities, the potential for these advancements to introduce new backdoor threats remains unexplored. This paper proposes Imperio, which harnesses the language understanding capabilities of NLP models to enrich backdoor attacks. Imperio provides a new model control experience. It empowers the adversary to control the victim model with arbitrary output through language-guided instructions. This is achieved using a language model to fuel a conditional trigger generator, with optimizations designed to extend its language understanding capabilities to backdoor instruction interpretation and execution. Our experiments across three datasets, five attacks, and nine defenses confirm Imperio's effectiveness. It can produce contextually adaptive triggers from text descriptions and control the victim model with desired outputs, even in scenarios not encountered during training. The attack maintains a high success rate across complex datasets without compromising the accuracy of clean inputs and also exhibits resilience against representative defenses. The source code is available at \url{https://khchow.com/Imperio}.



## **40. Safety and Performance, Why Not Both? Bi-Objective Optimized Model Compression against Heterogeneous Attacks Toward AI Software Deployment**

cs.AI

Accepted by IEEE Transactions on Software Engineering (TSE).  Camera-ready Version. arXiv admin note: substantial text overlap with  arXiv:2208.05969

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2401.00996v1) [paper-pdf](http://arxiv.org/pdf/2401.00996v1)

**Authors**: Jie Zhu, Leye Wang, Xiao Han, Anmin Liu, Tao Xie

**Abstract**: The size of deep learning models in artificial intelligence (AI) software is increasing rapidly, hindering the large-scale deployment on resource-restricted devices (e.g., smartphones). To mitigate this issue, AI software compression plays a crucial role, which aims to compress model size while keeping high performance. However, the intrinsic defects in a big model may be inherited by the compressed one. Such defects may be easily leveraged by adversaries, since a compressed model is usually deployed in a large number of devices without adequate protection. In this article, we aim to address the safe model compression problem from the perspective of safety-performance co-optimization. Specifically, inspired by the test-driven development (TDD) paradigm in software engineering, we propose a test-driven sparse training framework called SafeCompress. By simulating the attack mechanism as safety testing, SafeCompress can automatically compress a big model to a small one following the dynamic sparse training paradigm. Then, considering two kinds of representative and heterogeneous attack mechanisms, i.e., black-box membership inference attack and white-box membership inference attack, we develop two concrete instances called BMIA-SafeCompress and WMIA-SafeCompress. Further, we implement another instance called MMIA-SafeCompress by extending SafeCompress to defend against the occasion when adversaries conduct black-box and white-box membership inference attacks simultaneously. We conduct extensive experiments on five datasets for both computer vision and natural language processing tasks. The results show the effectiveness and generalizability of our framework. We also discuss how to adapt SafeCompress to other attacks besides membership inference attack, demonstrating the flexibility of SafeCompress.



## **41. Adversarial Item Promotion on Visually-Aware Recommender Systems by Guided Diffusion**

cs.IR

**SubmitDate**: 2024-01-02    [abs](http://arxiv.org/abs/2312.15826v3) [paper-pdf](http://arxiv.org/pdf/2312.15826v3)

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Guanhua Ye, Quoc Viet Hung Nguyen, Hongzhi Yin

**Abstract**: Visually-aware recommender systems have found widespread application in domains where visual elements significantly contribute to the inference of users' potential preferences. While the incorporation of visual information holds the promise of enhancing recommendation accuracy and alleviating the cold-start problem, it is essential to point out that the inclusion of item images may introduce substantial security challenges. Some existing works have shown that the item provider can manipulate item exposure rates to its advantage by constructing adversarial images. However, these works cannot reveal the real vulnerability of visually-aware recommender systems because (1) The generated adversarial images are markedly distorted, rendering them easily detectable by human observers; (2) The effectiveness of the attacks is inconsistent and even ineffective in some scenarios. To shed light on the real vulnerabilities of visually-aware recommender systems when confronted with adversarial images, this paper introduces a novel attack method, IPDGI (Item Promotion by Diffusion Generated Image). Specifically, IPDGI employs a guided diffusion model to generate adversarial samples designed to deceive visually-aware recommender systems. Taking advantage of accurately modeling benign images' distribution by diffusion models, the generated adversarial images have high fidelity with original images, ensuring the stealth of our IPDGI. To demonstrate the effectiveness of our proposed methods, we conduct extensive experiments on two commonly used e-commerce recommendation datasets (Amazon Beauty and Amazon Baby) with several typical visually-aware recommender systems. The experimental results show that our attack method has a significant improvement in both the performance of promoting the long-tailed (i.e., unpopular) items and the quality of generated adversarial images.



## **42. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

17 pages, 20 figures

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2310.10483v3) [paper-pdf](http://arxiv.org/pdf/2310.10483v3)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more practical attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging but practical scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves attack performance comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **43. Channel Reciprocity Attacks Using Intelligent Surfaces with Non-Diagonal Phase Shifts**

eess.SP

**SubmitDate**: 2024-01-01    [abs](http://arxiv.org/abs/2309.11665v2) [paper-pdf](http://arxiv.org/pdf/2309.11665v2)

**Authors**: Haoyu Wang, Zhu Han, A. Lee Swindlehurst

**Abstract**: While reconfigurable intelligent surface (RIS) technology has been shown to provide numerous benefits to wireless systems, in the hands of an adversary such technology can also be used to disrupt communication links. This paper describes and analyzes an RIS-based attack on multi-antenna wireless systems that operate in time-division duplex mode under the assumption of channel reciprocity. In particular, we show how an RIS with a non-diagonal (ND) phase shift matrix (referred to here as an ND-RIS) can be deployed to maliciously break the channel reciprocity and hence degrade the downlink network performance. Such an attack is entirely passive and difficult to detect and counteract. We provide a theoretical analysis of the degradation in the sum ergodic rate that results when an arbitrary malicious ND-RIS is deployed and design an approach based on the genetic algorithm for optimizing the ND structure under partial knowledge of the available channel state information. Our simulation results validate the analysis and demonstrate that an ND-RIS channel reciprocity attack can dramatically reduce the downlink throughput.



## **44. Is It Possible to Backdoor Face Forgery Detection with Natural Triggers?**

cs.CV

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2401.00414v1) [paper-pdf](http://arxiv.org/pdf/2401.00414v1)

**Authors**: Xiaoxuan Han, Songlin Yang, Wei Wang, Ziwen He, Jing Dong

**Abstract**: Deep neural networks have significantly improved the performance of face forgery detection models in discriminating Artificial Intelligent Generated Content (AIGC). However, their security is significantly threatened by the injection of triggers during model training (i.e., backdoor attacks). Although existing backdoor defenses and manual data selection can mitigate those using human-eye-sensitive triggers, such as patches or adversarial noises, the more challenging natural backdoor triggers remain insufficiently researched. To further investigate natural triggers, we propose a novel analysis-by-synthesis backdoor attack against face forgery detection models, which embeds natural triggers in the latent space. We thoroughly study such backdoor vulnerability from two perspectives: (1) Model Discrimination (Optimization-Based Trigger): we adopt a substitute detection model and find the trigger by minimizing the cross-entropy loss; (2) Data Distribution (Custom Trigger): we manipulate the uncommon facial attributes in the long-tailed distribution to generate poisoned samples without the supervision from detection models. Furthermore, to completely evaluate the detection models towards the latest AIGC, we utilize both state-of-the-art StyleGAN and Stable Diffusion for trigger generation. Finally, these backdoor triggers introduce specific semantic features to the generated poisoned samples (e.g., skin textures and smile), which are more natural and robust. Extensive experiments show that our method is superior from three levels: (1) Attack Success Rate: ours achieves a high attack success rate (over 99%) and incurs a small model accuracy drop (below 0.2%) with a low poisoning rate (less than 3%); (2) Backdoor Defense: ours shows better robust performance when faced with existing backdoor defense methods; (3) Human Inspection: ours is less human-eye-sensitive from a comprehensive user study.



## **45. Dictionary Attack on IMU-based Gait Authentication**

cs.CR

12 pages, 9 figures, accepted at AISec23 colocated with ACM CCS,  November 30, 2023, Copenhagen, Denmark

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2309.11766v2) [paper-pdf](http://arxiv.org/pdf/2309.11766v2)

**Authors**: Rajesh Kumar, Can Isik, Chilukuri K. Mohan

**Abstract**: We present a novel adversarial model for authentication systems that use gait patterns recorded by the inertial measurement unit (IMU) built into smartphones. The attack idea is inspired by and named after the concept of a dictionary attack on knowledge (PIN or password) based authentication systems. In particular, this work investigates whether it is possible to build a dictionary of IMUGait patterns and use it to launch an attack or find an imitator who can actively reproduce IMUGait patterns that match the target's IMUGait pattern. Nine physically and demographically diverse individuals walked at various levels of four predefined controllable and adaptable gait factors (speed, step length, step width, and thigh-lift), producing 178 unique IMUGait patterns. Each pattern attacked a wide variety of user authentication models. The deeper analysis of error rates (before and after the attack) challenges the belief that authentication systems based on IMUGait patterns are the most difficult to spoof; further research is needed on adversarial models and associated countermeasures.



## **46. Forbidden Facts: An Investigation of Competing Objectives in Llama-2**

cs.LG

Accepted to the ATTRIB and SoLaR workshops at NeurIPS 2023; (v3:  clarified experimental details)

**SubmitDate**: 2023-12-31    [abs](http://arxiv.org/abs/2312.08793v3) [paper-pdf](http://arxiv.org/pdf/2312.08793v3)

**Authors**: Tony T. Wang, Miles Wang, Kaivalya Hariharan, Nir Shavit

**Abstract**: LLMs often face competing pressures (for example helpfulness vs. harmlessness). To understand how models resolve such conflicts, we study Llama-2-chat models on the forbidden fact task. Specifically, we instruct Llama-2 to truthfully complete a factual recall statement while forbidding it from saying the correct answer. This often makes the model give incorrect answers. We decompose Llama-2 into 1000+ components, and rank each one with respect to how useful it is for forbidding the correct answer. We find that in aggregate, around 35 components are enough to reliably implement the full suppression behavior. However, these components are fairly heterogeneous and many operate using faulty heuristics. We discover that one of these heuristics can be exploited via a manually designed adversarial attack which we call The California Attack. Our results highlight some roadblocks standing in the way of being able to successfully interpret advanced ML systems. Project website available at https://forbiddenfacts.github.io .



## **47. Explainability-Driven Leaf Disease Classification using Adversarial Training and Knowledge Distillation**

cs.CV

10 pages, 8 figures, Accepted by ICAART 2024

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2401.00334v1) [paper-pdf](http://arxiv.org/pdf/2401.00334v1)

**Authors**: Sebastian-Vasile Echim, Iulian-Marius Tăiatu, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: This work focuses on plant leaf disease classification and explores three crucial aspects: adversarial training, model explainability, and model compression. The models' robustness against adversarial attacks is enhanced through adversarial training, ensuring accurate classification even in the presence of threats. Leveraging explainability techniques, we gain insights into the model's decision-making process, improving trust and transparency. Additionally, we explore model compression techniques to optimize computational efficiency while maintaining classification performance. Through our experiments, we determine that on a benchmark dataset, the robustness can be the price of the classification accuracy with performance reductions of 3%-20% for regular tests and gains of 50%-70% for adversarial attack tests. We also demonstrate that a student model can be 15-25 times more computationally efficient for a slight performance reduction, distilling the knowledge of more complex models.



## **48. Unraveling the Connections between Privacy and Certified Robustness in Federated Learning Against Poisoning Attacks**

cs.CR

ACM CCS 2023

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2209.04030v3) [paper-pdf](http://arxiv.org/pdf/2209.04030v3)

**Authors**: Chulin Xie, Yunhui Long, Pin-Yu Chen, Qinbin Li, Arash Nourian, Sanmi Koyejo, Bo Li

**Abstract**: Federated learning (FL) provides an efficient paradigm to jointly train a global model leveraging data from distributed users. As local training data comes from different users who may not be trustworthy, several studies have shown that FL is vulnerable to poisoning attacks. Meanwhile, to protect the privacy of local users, FL is usually trained in a differentially private way (DPFL). Thus, in this paper, we ask: What are the underlying connections between differential privacy and certified robustness in FL against poisoning attacks? Can we leverage the innate privacy property of DPFL to provide certified robustness for FL? Can we further improve the privacy of FL to improve such robustness certification? We first investigate both user-level and instance-level privacy of FL and provide formal privacy analysis to achieve improved instance-level privacy. We then provide two robustness certification criteria: certified prediction and certified attack inefficacy for DPFL on both user and instance levels. Theoretically, we provide the certified robustness of DPFL based on both criteria given a bounded number of adversarial users or instances. Empirically, we conduct extensive experiments to verify our theories under a range of poisoning attacks on different datasets. We find that increasing the level of privacy protection in DPFL results in stronger certified attack inefficacy; however, it does not necessarily lead to a stronger certified prediction. Thus, achieving the optimal certified prediction requires a proper balance between privacy and utility loss.



## **49. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

cs.CV

18 pages, 13 figures

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2305.17939v2) [paper-pdf](http://arxiv.org/pdf/2305.17939v2)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.



## **50. ReMAV: Reward Modeling of Autonomous Vehicles for Finding Likely Failure Events**

cs.AI

**SubmitDate**: 2023-12-30    [abs](http://arxiv.org/abs/2308.14550v2) [paper-pdf](http://arxiv.org/pdf/2308.14550v2)

**Authors**: Aizaz Sharif, Dusica Marijan

**Abstract**: Autonomous vehicles are advanced driving systems that are well known to be vulnerable to various adversarial attacks, compromising vehicle safety and posing a risk to other road users. Rather than actively training complex adversaries by interacting with the environment, there is a need to first intelligently find and reduce the search space to only those states where autonomous vehicles are found to be less confident. In this paper, we propose a black-box testing framework ReMAV that uses offline trajectories first to analyze the existing behavior of autonomous vehicles and determine appropriate thresholds to find the probability of failure events. To this end, we introduce a three-step methodology which i) uses offline state action pairs of any autonomous vehicle under test, ii) builds an abstract behavior representation using our designed reward modeling technique to analyze states with uncertain driving decisions, and iii) uses a disturbance model for minimal perturbation attacks where the driving decisions are less confident. Our reward modeling technique helps in creating a behavior representation that allows us to highlight regions of likely uncertain behavior even when the standard autonomous vehicle performs well. We perform our experiments in a high-fidelity urban driving environment using three different driving scenarios containing single- and multi-agent interactions. Our experiment shows an increase in 35, 23, 48, and 50% in the occurrences of vehicle collision, road object collision, pedestrian collision, and offroad steering events, respectively by the autonomous vehicle under test, demonstrating a significant increase in failure events. We compare ReMAV with two baselines and show that ReMAV demonstrates significantly better effectiveness in generating failure events compared to the baselines in all evaluation metrics.



