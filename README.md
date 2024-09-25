# Latest Adversarial Attack Papers
**update at 2024-09-25 10:24:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Order of Magnitude Speedups for LLM Membership Inference**

cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14513v2) [paper-pdf](http://arxiv.org/pdf/2409.14513v2)

**Authors**: Rongting Zhang, Martin Bertran, Aaron Roth

**Abstract**: Large Language Models (LLMs) have the promise to revolutionize computing broadly, but their complexity and extensive training data also expose significant privacy vulnerabilities. One of the simplest privacy risks associated with LLMs is their susceptibility to membership inference attacks (MIAs), wherein an adversary aims to determine whether a specific data point was part of the model's training set. Although this is a known risk, state of the art methodologies for MIAs rely on training multiple computationally costly shadow models, making risk evaluation prohibitive for large models. Here we adapt a recent line of work which uses quantile regression to mount membership inference attacks; we extend this work by proposing a low-cost MIA that leverages an ensemble of small quantile regression models to determine if a document belongs to the model's training set or not. We demonstrate the effectiveness of this approach on fine-tuned LLMs of varying families (OPT, Pythia, Llama) and across multiple datasets. Across all scenarios we obtain comparable or improved accuracy compared to state of the art shadow model approaches, with as little as 6% of their computation budget. We demonstrate increased effectiveness across multi-epoch trained target models, and architecture miss-specification robustness, that is, we can mount an effective attack against a model using a different tokenizer and architecture, without requiring knowledge on the target model.



## **2. Unclonable Non-Interactive Zero-Knowledge**

cs.CR

Clarified definitions and included applications

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2310.07118v3) [paper-pdf](http://arxiv.org/pdf/2310.07118v3)

**Authors**: Ruta Jawale, Dakshita Khurana

**Abstract**: A non-interactive ZK (NIZK) proof enables verification of NP statements without revealing secrets about them. However, an adversary that obtains a NIZK proof may be able to clone this proof and distribute arbitrarily many copies of it to various entities: this is inevitable for any proof that takes the form of a classical string. In this paper, we ask whether it is possible to rely on quantum information in order to build NIZK proof systems that are impossible to clone.   We define and construct unclonable non-interactive zero-knowledge arguments (of knowledge) for NP, addressing a question first posed by Aaronson (CCC 2009). Besides satisfying the zero-knowledge and argument of knowledge properties, these proofs additionally satisfy unclonability. Very roughly, this ensures that no adversary can split an honestly generated proof of membership of an instance $x$ in an NP language $\mathcal{L}$ and distribute copies to multiple entities that all obtain accepting proofs of membership of $x$ in $\mathcal{L}$. Our result has applications to unclonable signatures of knowledge, which we define and construct in this work; these non-interactively prevent replay attacks.



## **3. Cyber Knowledge Completion Using Large Language Models**

cs.CR

7 pages, 2 figures. Submitted to 2024 IEEE International Conference  on Big Data

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16176v1) [paper-pdf](http://arxiv.org/pdf/2409.16176v1)

**Authors**: Braden K Webb, Sumit Purohit, Rounak Meyur

**Abstract**: The integration of the Internet of Things (IoT) into Cyber-Physical Systems (CPSs) has expanded their cyber-attack surface, introducing new and sophisticated threats with potential to exploit emerging vulnerabilities. Assessing the risks of CPSs is increasingly difficult due to incomplete and outdated cybersecurity knowledge. This highlights the urgent need for better-informed risk assessments and mitigation strategies. While previous efforts have relied on rule-based natural language processing (NLP) tools to map vulnerabilities, weaknesses, and attack patterns, recent advancements in Large Language Models (LLMs) present a unique opportunity to enhance cyber-attack knowledge completion through improved reasoning, inference, and summarization capabilities. We apply embedding models to encapsulate information on attack patterns and adversarial techniques, generating mappings between them using vector embeddings. Additionally, we propose a Retrieval-Augmented Generation (RAG)-based approach that leverages pre-trained models to create structured mappings between different taxonomies of threat patterns. Further, we use a small hand-labeled dataset to compare the proposed RAG-based approach to a baseline standard binary classification model. Thus, the proposed approach provides a comprehensive framework to address the challenge of cyber-attack knowledge graph completion.



## **4. A Strong Separation for Adversarially Robust $\ell_0$ Estimation for Linear Sketches**

cs.DS

FOCS 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16153v1) [paper-pdf](http://arxiv.org/pdf/2409.16153v1)

**Authors**: Elena Gribelyuk, Honghao Lin, David P. Woodruff, Huacheng Yu, Samson Zhou

**Abstract**: The majority of streaming problems are defined and analyzed in a static setting, where the data stream is any worst-case sequence of insertions and deletions that is fixed in advance. However, many real-world applications require a more flexible model, where an adaptive adversary may select future stream elements after observing the previous outputs of the algorithm. Over the last few years, there has been increased interest in proving lower bounds for natural problems in the adaptive streaming model. In this work, we give the first known adaptive attack against linear sketches for the well-studied $\ell_0$-estimation problem over turnstile, integer streams. For any linear streaming algorithm $\mathcal{A}$ that uses sketching matrix $\mathbf{A}\in \mathbb{Z}^{r \times n}$ where $n$ is the size of the universe, this attack makes $\tilde{\mathcal{O}}(r^8)$ queries and succeeds with high constant probability in breaking the sketch. We also give an adaptive attack against linear sketches for the $\ell_0$-estimation problem over finite fields $\mathbb{F}_p$, which requires a smaller number of $\tilde{\mathcal{O}}(r^3)$ queries. Finally, we provide an adaptive attack over $\mathbb{R}^n$ against linear sketches $\mathbf{A} \in \mathbb{R}^{r \times n}$ for $\ell_0$-estimation, in the setting where $\mathbf{A}$ has all nonzero subdeterminants at least $\frac{1}{\textrm{poly}(r)}$. Our results provide an exponential improvement over the previous number of queries known to break an $\ell_0$-estimation sketch.



## **5. Scenario of Use Scheme: Threat Model Specification for Speaker Privacy Protection in the Medical Domain**

eess.AS

Accepted and published at SPSC Symposium 2024 4th Symposium on  Security and Privacy in Speech Communication. Interspeech 2024

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16106v1) [paper-pdf](http://arxiv.org/pdf/2409.16106v1)

**Authors**: Mehtab Ur Rahman, Martha Larson, Louis ten Bosch, Cristian Tejedor-García

**Abstract**: Speech recordings are being more frequently used to detect and monitor disease, leading to privacy concerns. Beyond cryptography, protection of speech can be addressed by approaches, such as perturbation, disentanglement, and re-synthesis, that eliminate sensitive information of the speaker, leaving the information necessary for medical analysis purposes. In order for such privacy protective approaches to be developed, clear and systematic specifications of assumptions concerning medical settings and the needs of medical professionals are necessary. In this paper, we propose a Scenario of Use Scheme that incorporates an Attacker Model, which characterizes the adversary against whom the speaker's privacy must be defended, and a Protector Model, which specifies the defense. We discuss the connection of the scheme with previous work on speech privacy. Finally, we present a concrete example of a specified Scenario of Use and a set of experiments about protecting speaker data against gender inference attacks while maintaining utility for Parkinson's detection.



## **6. Adversarial Attacks on Machine Learning-Aided Visualizations**

cs.CR

This version of the article has been accepted for publication, after  peer review (when applicable) but is not the Version of Record and does not  reflect post-acceptance improvements, or any corrections. The Version of  Record is available online at: http://dx.doi.org/10.1007/s12650-024-01029-2

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.02485v2) [paper-pdf](http://arxiv.org/pdf/2409.02485v2)

**Authors**: Takanori Fujiwara, Kostiantyn Kucher, Junpeng Wang, Rafael M. Martins, Andreas Kerren, Anders Ynnerman

**Abstract**: Research in ML4VIS investigates how to use machine learning (ML) techniques to generate visualizations, and the field is rapidly growing with high societal impact. However, as with any computational pipeline that employs ML processes, ML4VIS approaches are susceptible to a range of ML-specific adversarial attacks. These attacks can manipulate visualization generations, causing analysts to be tricked and their judgments to be impaired. Due to a lack of synthesis from both visualization and ML perspectives, this security aspect is largely overlooked by the current ML4VIS literature. To bridge this gap, we investigate the potential vulnerabilities of ML-aided visualizations from adversarial attacks using a holistic lens of both visualization and ML perspectives. We first identify the attack surface (i.e., attack entry points) that is unique in ML-aided visualizations. We then exemplify five different adversarial attacks. These examples highlight the range of possible attacks when considering the attack surface and multiple different adversary capabilities. Our results show that adversaries can induce various attacks, such as creating arbitrary and deceptive visualizations, by systematically identifying input attributes that are influential in ML inferences. Based on our observations of the attack surface characteristics and the attack examples, we underline the importance of comprehensive studies of security issues and defense mechanisms as a call of urgency for the ML4VIS community.



## **7. When Witnesses Defend: A Witness Graph Topological Layer for Adversarial Graph Learning**

cs.LG

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14161v2) [paper-pdf](http://arxiv.org/pdf/2409.14161v2)

**Authors**: Naheed Anjum Arafat, Debabrota Basu, Yulia Gel, Yuzhou Chen

**Abstract**: Capitalizing on the intuitive premise that shape characteristics are more robust to perturbations, we bridge adversarial graph learning with the emerging tools from computational topology, namely, persistent homology representations of graphs. We introduce the concept of witness complex to adversarial analysis on graphs, which allows us to focus only on the salient shape characteristics of graphs, yielded by the subset of the most essential nodes (i.e., landmarks), with minimal loss of topological information on the whole graph. The remaining nodes are then used as witnesses, governing which higher-order graph substructures are incorporated into the learning process. Armed with the witness mechanism, we design Witness Graph Topological Layer (WGTL), which systematically integrates both local and global topological graph feature representations, the impact of which is, in turn, automatically controlled by the robust regularized topological loss. Given the attacker's budget, we derive the important stability guarantees of both local and global topology encodings and the associated robust topological loss. We illustrate the versatility and efficiency of WGTL by its integration with five GNNs and three existing non-topological defense mechanisms. Our extensive experiments across six datasets demonstrate that WGTL boosts the robustness of GNNs across a range of perturbations and against a range of adversarial attacks, leading to relative gains of up to 18%.



## **8. Adversarial Watermarking for Face Recognition**

cs.CV

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.16056v1) [paper-pdf](http://arxiv.org/pdf/2409.16056v1)

**Authors**: Yuguang Yao, Anil Jain, Sijia Liu

**Abstract**: Watermarking is an essential technique for embedding an identifier (i.e., watermark message) within digital images to assert ownership and monitor unauthorized alterations. In face recognition systems, watermarking plays a pivotal role in ensuring data integrity and security. However, an adversary could potentially interfere with the watermarking process, significantly impairing recognition performance. We explore the interaction between watermarking and adversarial attacks on face recognition models. Our findings reveal that while watermarking or input-level perturbation alone may have a negligible effect on recognition accuracy, the combined effect of watermarking and perturbation can result in an adversarial watermarking attack, significantly degrading recognition performance. Specifically, we introduce a novel threat model, the adversarial watermarking attack, which remains stealthy in the absence of watermarking, allowing images to be correctly recognized initially. However, once watermarking is applied, the attack is activated, causing recognition failures. Our study reveals a previously unrecognized vulnerability: adversarial perturbations can exploit the watermark message to evade face recognition systems. Evaluated on the CASIA-WebFace dataset, our proposed adversarial watermarking attack reduces face matching accuracy by 67.2% with an $\ell_\infty$ norm-measured perturbation strength of ${2}/{255}$ and by 95.9% with a strength of ${4}/{255}$.



## **9. Adversarial Backdoor Defense in CLIP**

cs.CV

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15968v1) [paper-pdf](http://arxiv.org/pdf/2409.15968v1)

**Authors**: Junhao Kuang, Siyuan Liang, Jiawei Liang, Kuanrong Liu, Xiaochun Cao

**Abstract**: Multimodal contrastive pretraining, exemplified by models like CLIP, has been found to be vulnerable to backdoor attacks. While current backdoor defense methods primarily employ conventional data augmentation to create augmented samples aimed at feature alignment, these methods fail to capture the distinct features of backdoor samples, resulting in suboptimal defense performance. Observations reveal that adversarial examples and backdoor samples exhibit similarities in the feature space within the compromised models. Building on this insight, we propose Adversarial Backdoor Defense (ABD), a novel data augmentation strategy that aligns features with meticulously crafted adversarial examples. This approach effectively disrupts the backdoor association. Our experiments demonstrate that ABD provides robust defense against both traditional uni-modal and multimodal backdoor attacks targeting CLIP. Compared to the current state-of-the-art defense method, CleanCLIP, ABD reduces the attack success rate by 8.66% for BadNet, 10.52% for Blended, and 53.64% for BadCLIP, while maintaining a minimal average decrease of just 1.73% in clean accuracy.



## **10. Can Go AIs be adversarially robust?**

cs.LG

59 pages

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2406.12843v2) [paper-pdf](http://arxiv.org/pdf/2406.12843v2)

**Authors**: Tom Tseng, Euan McLean, Kellin Pelrine, Tony T. Wang, Adam Gleave

**Abstract**: Prior work found that superhuman Go AIs can be defeated by simple adversarial strategies, especially "cyclic" attacks. In this paper, we study whether adding natural countermeasures can achieve robustness in Go, a favorable domain for robustness since it benefits from incredible average-case capability and a narrow, innately adversarial setting. We test three defenses: adversarial training on hand-constructed positions, iterated adversarial training, and changing the network architecture. We find that though some of these defenses protect against previously discovered attacks, none withstand freshly trained adversaries. Furthermore, most of the reliably effective attacks these adversaries discover are different realizations of the same overall class of cyclic attacks. Our results suggest that building robust AI systems is challenging even with extremely superhuman systems in some of the most tractable settings, and highlight two key gaps: efficient generalization in defenses, and diversity in training. For interactive examples of attacks and a link to our codebase, see https://goattack.far.ai.



## **11. Goal-guided Generative Prompt Injection Attack on Large Language Models**

cs.CR

11 pages, 6 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2404.07234v3) [paper-pdf](http://arxiv.org/pdf/2404.07234v3)

**Authors**: Chong Zhang, Mingyu Jin, Qinkai Yu, Chengzhi Liu, Haochen Xue, Xiaobo Jin

**Abstract**: Current large language models (LLMs) provide a strong foundation for large-scale user-oriented natural language tasks. A large number of users can easily inject adversarial text or instructions through the user interface, thus causing LLMs model security challenges. Although there is currently a large amount of research on prompt injection attacks, most of these black-box attacks use heuristic strategies. It is unclear how these heuristic strategies relate to the success rate of attacks and thus effectively improve model robustness. To solve this problem, we redefine the goal of the attack: to maximize the KL divergence between the conditional probabilities of the clean text and the adversarial text. Furthermore, we prove that maximizing the KL divergence is equivalent to maximizing the Mahalanobis distance between the embedded representation $x$ and $x'$ of the clean text and the adversarial text when the conditional probability is a Gaussian distribution and gives a quantitative relationship on $x$ and $x'$. Then we designed a simple and effective goal-guided generative prompt injection strategy (G2PIA) to find an injection text that satisfies specific constraints to achieve the optimal attack effect approximately. It is particularly noteworthy that our attack method is a query-free black-box attack method with low computational cost. Experimental results on seven LLM models and four datasets show the effectiveness of our attack method.



## **12. Efficient and Effective Model Extraction**

cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.14122v2) [paper-pdf](http://arxiv.org/pdf/2409.14122v2)

**Authors**: Hongyu Zhu, Wentao Hu, Sichu Liang, Fangqi Li, Wenwen Wang, Shilin Wang

**Abstract**: Model extraction aims to create a functionally similar copy from a machine learning as a service (MLaaS) API with minimal overhead, typically for illicit profit or as a precursor to further attacks, posing a significant threat to the MLaaS ecosystem. However, recent studies have shown that model extraction is highly inefficient, particularly when the target task distribution is unavailable. In such cases, even substantially increasing the attack budget fails to produce a sufficiently similar replica, reducing the adversary's motivation to pursue extraction attacks. In this paper, we revisit the elementary design choices throughout the extraction lifecycle. We propose an embarrassingly simple yet dramatically effective algorithm, Efficient and Effective Model Extraction (E3), focusing on both query preparation and training routine. E3 achieves superior generalization compared to state-of-the-art methods while minimizing computational costs. For instance, with only 0.005 times the query budget and less than 0.2 times the runtime, E3 outperforms classical generative model based data-free model extraction by an absolute accuracy improvement of over 50% on CIFAR-10. Our findings underscore the persistent threat posed by model extraction and suggest that it could serve as a valuable benchmarking algorithm for future security evaluations.



## **13. Toward Mixture-of-Experts Enabled Trustworthy Semantic Communication for 6G Networks**

cs.NI

8 pages, 3 figures

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15695v1) [paper-pdf](http://arxiv.org/pdf/2409.15695v1)

**Authors**: Jiayi He, Xiaofeng Luo, Jiawen Kang, Hongyang Du, Zehui Xiong, Ci Chen, Dusit Niyato, Xuemin Shen

**Abstract**: Semantic Communication (SemCom) plays a pivotal role in 6G networks, offering a viable solution for future efficient communication. Deep Learning (DL)-based semantic codecs further enhance this efficiency. However, the vulnerability of DL models to security threats, such as adversarial attacks, poses significant challenges for practical applications of SemCom systems. These vulnerabilities enable attackers to tamper with messages and eavesdrop on private information, especially in wireless communication scenarios. Although existing defenses attempt to address specific threats, they often fail to simultaneously handle multiple heterogeneous attacks. To overcome this limitation, we introduce a novel Mixture-of-Experts (MoE)-based SemCom system. This system comprises a gating network and multiple experts, each specializing in different security challenges. The gating network adaptively selects suitable experts to counter heterogeneous attacks based on user-defined security requirements. Multiple experts collaborate to accomplish semantic communication tasks while meeting the security requirements of users. A case study in vehicular networks demonstrates the efficacy of the MoE-based SemCom system. Simulation results show that the proposed MoE-based SemCom system effectively mitigates concurrent heterogeneous attacks, with minimal impact on downstream task accuracy.



## **14. Data Poisoning-based Backdoor Attack Framework against Supervised Learning Rules of Spiking Neural Networks**

cs.CR

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.15670v1) [paper-pdf](http://arxiv.org/pdf/2409.15670v1)

**Authors**: Lingxin Jin, Meiyu Lin, Wei Jiang, Jinyu Zhan

**Abstract**: Spiking Neural Networks (SNNs), the third generation neural networks, are known for their low energy consumption and high robustness. SNNs are developing rapidly and can compete with Artificial Neural Networks (ANNs) in many fields. To ensure that the widespread use of SNNs does not cause serious security incidents, much research has been conducted to explore the robustness of SNNs under adversarial sample attacks. However, many other unassessed security threats exist, such as highly stealthy backdoor attacks. Therefore, to fill the research gap in this and further explore the security vulnerabilities of SNNs, this paper explores the robustness performance of SNNs trained by supervised learning rules under backdoor attacks. Specifically, the work herein includes: i) We propose a generic backdoor attack framework that can be launched against the training process of existing supervised learning rules and covers all learnable dataset types of SNNs. ii) We analyze the robustness differences between different learning rules and between SNN and ANN, which suggests that SNN no longer has inherent robustness under backdoor attacks. iii) We reveal the vulnerability of conversion-dependent learning rules caused by backdoor migration and further analyze the migration ability during the conversion process, finding that the backdoor migration rate can even exceed 99%. iv) Finally, we discuss potential countermeasures against this kind of backdoor attack and its technical challenges and point out several promising research directions.



## **15. Adversarial Attacks to Multi-Modal Models**

cs.CR

To appear in the ACM Workshop on Large AI Systems and Models with  Privacy and Safety Analysis 2024 (LAMPS '24)

**SubmitDate**: 2024-09-24    [abs](http://arxiv.org/abs/2409.06793v2) [paper-pdf](http://arxiv.org/pdf/2409.06793v2)

**Authors**: Zhihao Dou, Xin Hu, Haibo Yang, Zhuqing Liu, Minghong Fang

**Abstract**: Multi-modal models have gained significant attention due to their powerful capabilities. These models effectively align embeddings across diverse data modalities, showcasing superior performance in downstream tasks compared to their unimodal counterparts. Recent study showed that the attacker can manipulate an image or audio file by altering it in such a way that its embedding matches that of an attacker-chosen targeted input, thereby deceiving downstream models. However, this method often underperforms due to inherent disparities in data from different modalities. In this paper, we introduce CrossFire, an innovative approach to attack multi-modal models. CrossFire begins by transforming the targeted input chosen by the attacker into a format that matches the modality of the original image or audio file. We then formulate our attack as an optimization problem, aiming to minimize the angular deviation between the embeddings of the transformed input and the modified image or audio file. Solving this problem determines the perturbations to be added to the original media. Our extensive experiments on six real-world benchmark datasets reveal that CrossFire can significantly manipulate downstream tasks, surpassing existing attacks. Additionally, we evaluate six defensive strategies against CrossFire, finding that current defenses are insufficient to counteract our CrossFire.



## **16. Interpretability-Guided Test-Time Adversarial Defense**

cs.CV

ECCV 2024. Project Page:  https://lilywenglab.github.io/Interpretability-Guided-Defense/

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15190v1) [paper-pdf](http://arxiv.org/pdf/2409.15190v1)

**Authors**: Akshay Kulkarni, Tsui-Wei Weng

**Abstract**: We propose a novel and low-cost test-time adversarial defense by devising interpretability-guided neuron importance ranking methods to identify neurons important to the output classes. Our method is a training-free approach that can significantly improve the robustness-accuracy tradeoff while incurring minimal computational overhead. While being among the most efficient test-time defenses (4x faster), our method is also robust to a wide range of black-box, white-box, and adaptive attacks that break previous test-time defenses. We demonstrate the efficacy of our method for CIFAR10, CIFAR100, and ImageNet-1k on the standard RobustBench benchmark (with average gains of 2.6%, 4.9%, and 2.8% respectively). We also show improvements (average 1.5%) over the state-of-the-art test-time defenses even under strong adaptive attacks.



## **17. Log-normal Mutations and their Use in Detecting Surreptitious Fake Images**

cs.AI

log-normal mutations and their use in detecting surreptitious fake  images

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15119v1) [paper-pdf](http://arxiv.org/pdf/2409.15119v1)

**Authors**: Ismail Labiad, Thomas Bäck, Pierre Fernandez, Laurent Najman, Tom Sanders, Furong Ye, Mariia Zameshina, Olivier Teytaud

**Abstract**: In many cases, adversarial attacks are based on specialized algorithms specifically dedicated to attacking automatic image classifiers. These algorithms perform well, thanks to an excellent ad hoc distribution of initial attacks. However, these attacks are easily detected due to their specific initial distribution. We therefore consider other black-box attacks, inspired from generic black-box optimization tools, and in particular the log-normal algorithm.   We apply the log-normal method to the attack of fake detectors, and get successful attacks: importantly, these attacks are not detected by detectors specialized on classical adversarial attacks. Then, combining these attacks and deep detection, we create improved fake detectors.



## **18. SHFL: Secure Hierarchical Federated Learning Framework for Edge Networks**

cs.LG

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15067v1) [paper-pdf](http://arxiv.org/pdf/2409.15067v1)

**Authors**: Omid Tavallaie, Kanchana Thilakarathna, Suranga Seneviratne, Aruna Seneviratne, Albert Y. Zomaya

**Abstract**: Federated Learning (FL) is a distributed machine learning paradigm designed for privacy-sensitive applications that run on resource-constrained devices with non-Identically and Independently Distributed (IID) data. Traditional FL frameworks adopt the client-server model with a single-level aggregation (AGR) process, where the server builds the global model by aggregating all trained local models received from client devices. However, this conventional approach encounters challenges, including susceptibility to model/data poisoning attacks. In recent years, advancements in the Internet of Things (IoT) and edge computing have enabled the development of hierarchical FL systems with a two-level AGR process running at edge and cloud servers. In this paper, we propose a Secure Hierarchical FL (SHFL) framework to address poisoning attacks in hierarchical edge networks. By aggregating trained models at the edge, SHFL employs two novel methods to address model/data poisoning attacks in the presence of client adversaries: 1) a client selection algorithm running at the edge for choosing IoT devices to participate in training, and 2) a model AGR method designed based on convex optimization theory to reduce the impact of edge models from networks with adversaries in the process of computing the global model (at the cloud level). The evaluation results reveal that compared to state-of-the-art methods, SHFL significantly increases the maximum accuracy achieved by the global model in the presence of client adversaries applying model/data poisoning attacks.



## **19. Improving Adversarial Robustness for 3D Point Cloud Recognition at Test-Time through Purified Self-Training**

cs.CV

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.14940v1) [paper-pdf](http://arxiv.org/pdf/2409.14940v1)

**Authors**: Jinpeng Lin, Xulei Yang, Tianrui Li, Xun Xu

**Abstract**: Recognizing 3D point cloud plays a pivotal role in many real-world applications. However, deploying 3D point cloud deep learning model is vulnerable to adversarial attacks. Despite many efforts into developing robust model by adversarial training, they may become less effective against emerging attacks. This limitation motivates the development of adversarial purification which employs generative model to mitigate the impact of adversarial attacks. In this work, we highlight the remaining challenges from two perspectives. First, the purification based method requires retraining the classifier on purified samples which introduces additional computation overhead. Moreover, in a more realistic scenario, testing samples arrives in a streaming fashion and adversarial samples are not isolated from clean samples. These challenges motivates us to explore dynamically update model upon observing testing samples. We proposed a test-time purified self-training strategy to achieve this objective. Adaptive thresholding and feature distribution alignment are introduced to improve the robustness of self-training. Extensive results on different adversarial attacks suggest the proposed method is complementary to purification based method in handling continually changing adversarial attacks on the testing data stream.



## **20. Attack Atlas: A Practitioner's Perspective on Challenges and Pitfalls in Red Teaming GenAI**

cs.CR

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.15398v1) [paper-pdf](http://arxiv.org/pdf/2409.15398v1)

**Authors**: Ambrish Rawat, Stefan Schoepf, Giulio Zizzo, Giandomenico Cornacchia, Muhammad Zaid Hameed, Kieran Fraser, Erik Miehling, Beat Buesser, Elizabeth M. Daly, Mark Purcell, Prasanna Sattigeri, Pin-Yu Chen, Kush R. Varshney

**Abstract**: As generative AI, particularly large language models (LLMs), become increasingly integrated into production applications, new attack surfaces and vulnerabilities emerge and put a focus on adversarial threats in natural language and multi-modal systems. Red-teaming has gained importance in proactively identifying weaknesses in these systems, while blue-teaming works to protect against such adversarial attacks. Despite growing academic interest in adversarial risks for generative AI, there is limited guidance tailored for practitioners to assess and mitigate these challenges in real-world environments. To address this, our contributions include: (1) a practical examination of red- and blue-teaming strategies for securing generative AI, (2) identification of key challenges and open questions in defense development and evaluation, and (3) the Attack Atlas, an intuitive framework that brings a practical approach to analyzing single-turn input attacks, placing it at the forefront for practitioners. This work aims to bridge the gap between academic insights and practical security measures for the protection of generative AI systems.



## **21. Transfer-based Adversarial Poisoning Attacks for Online (MIMO-)Deep Receviers**

eess.SP

15 pages, 14 figures

**SubmitDate**: 2024-09-23    [abs](http://arxiv.org/abs/2409.02430v3) [paper-pdf](http://arxiv.org/pdf/2409.02430v3)

**Authors**: Kunze Wu, Weiheng Jiang, Dusit Niyato, Yinghuan Li, Chuang Luo

**Abstract**: Recently, the design of wireless receivers using deep neural networks (DNNs), known as deep receivers, has attracted extensive attention for ensuring reliable communication in complex channel environments. To adapt quickly to dynamic channels, online learning has been adopted to update the weights of deep receivers with over-the-air data (e.g., pilots). However, the fragility of neural models and the openness of wireless channels expose these systems to malicious attacks. To this end, understanding these attack methods is essential for robust receiver design. In this paper, we propose a transfer-based adversarial poisoning attack method for online receivers. Without knowledge of the attack target, adversarial perturbations are injected to the pilots, poisoning the online deep receiver and impairing its ability to adapt to dynamic channels and nonlinear effects. In particular, our attack method targets Deep Soft Interference Cancellation (DeepSIC)[1] using online meta-learning. As a classical model-driven deep receiver, DeepSIC incorporates wireless domain knowledge into its architecture. This integration allows it to adapt efficiently to time-varying channels with only a small number of pilots, achieving optimal performance in a multi-input and multi-output (MIMO) scenario. The deep receiver in this scenario has a number of applications in the field of wireless communication, which motivates our study of the attack methods targeting it. Specifically, we demonstrate the effectiveness of our attack in simulations on synthetic linear, synthetic nonlinear, static, and COST 2100 channels. Simulation results indicate that the proposed poisoning attack significantly reduces the performance of online receivers in rapidly changing scenarios.



## **22. Backtracking Improves Generation Safety**

cs.LG

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14586v1) [paper-pdf](http://arxiv.org/pdf/2409.14586v1)

**Authors**: Yiming Zhang, Jianfeng Chi, Hailey Nguyen, Kartikeya Upasani, Daniel M. Bikel, Jason Weston, Eric Michael Smith

**Abstract**: Text generation has a fundamental limitation almost by definition: there is no taking back tokens that have been generated, even when they are clearly problematic. In the context of language model safety, when a partial unsafe generation is produced, language models by their nature tend to happily keep on generating similarly unsafe additional text. This is in fact how safety alignment of frontier models gets circumvented in the wild, despite great efforts in improving their safety. Deviating from the paradigm of approaching safety alignment as prevention (decreasing the probability of harmful responses), we propose backtracking, a technique that allows language models to "undo" and recover from their own unsafe generation through the introduction of a special [RESET] token. Our method can be incorporated into either SFT or DPO training to optimize helpfulness and harmlessness. We show that models trained to backtrack are consistently safer than baseline models: backtracking Llama-3-8B is four times more safe than the baseline model (6.1\% $\to$ 1.5\%) in our evaluations without regression in helpfulness. Our method additionally provides protection against four adversarial attacks including an adaptive attack, despite not being trained to do so.



## **23. Enhancing LLM-based Autonomous Driving Agents to Mitigate Perception Attacks**

cs.CR

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2409.14488v1) [paper-pdf](http://arxiv.org/pdf/2409.14488v1)

**Authors**: Ruoyu Song, Muslum Ozgur Ozmen, Hyungsub Kim, Antonio Bianchi, Z. Berkay Celik

**Abstract**: There is a growing interest in integrating Large Language Models (LLMs) with autonomous driving (AD) systems. However, AD systems are vulnerable to attacks against their object detection and tracking (ODT) functions. Unfortunately, our evaluation of four recent LLM agents against ODT attacks shows that the attacks are 63.26% successful in causing them to crash or violate traffic rules due to (1) misleading memory modules that provide past experiences for decision making, (2) limitations of prompts in identifying inconsistencies, and (3) reliance on ground truth perception data.   In this paper, we introduce Hudson, a driving reasoning agent that extends prior LLM-based driving systems to enable safer decision making during perception attacks while maintaining effectiveness under benign conditions. Hudson achieves this by first instrumenting the AD software to collect real-time perception results and contextual information from the driving scene. This data is then formalized into a domain-specific language (DSL). To guide the LLM in detecting and making safe control decisions during ODT attacks, Hudson translates the DSL into natural language, along with a list of custom attack detection instructions. Following query execution, Hudson analyzes the LLM's control decision to understand its causal reasoning process.   We evaluate the effectiveness of Hudson using a proprietary LLM (GPT-4) and two open-source LLMs (Llama and Gemma) in various adversarial driving scenarios. GPT-4, Llama, and Gemma achieve, on average, an attack detection accuracy of 83. 3%, 63. 6%, and 73. 6%. Consequently, they make safe control decisions in 86.4%, 73.9%, and 80% of the attacks. Our results, following the growing interest in integrating LLMs into AD systems, highlight the strengths of LLMs and their potential to detect and mitigate ODT attacks.



## **24. Stop Reasoning! When Multimodal LLM with Chain-of-Thought Reasoning Meets Adversarial Image**

cs.CV

**SubmitDate**: 2024-09-22    [abs](http://arxiv.org/abs/2402.14899v3) [paper-pdf](http://arxiv.org/pdf/2402.14899v3)

**Authors**: Zefeng Wang, Zhen Han, Shuo Chen, Fan Xue, Zifeng Ding, Xun Xiao, Volker Tresp, Philip Torr, Jindong Gu

**Abstract**: Multimodal LLMs (MLLMs) with a great ability of text and image understanding have received great attention. To achieve better reasoning with MLLMs, Chain-of-Thought (CoT) reasoning has been widely explored, which further promotes MLLMs' explainability by giving intermediate reasoning steps. Despite the strong power demonstrated by MLLMs in multimodal reasoning, recent studies show that MLLMs still suffer from adversarial images. This raises the following open questions: Does CoT also enhance the adversarial robustness of MLLMs? What do the intermediate reasoning steps of CoT entail under adversarial attacks? To answer these questions, we first generalize existing attacks to CoT-based inferences by attacking the two main components, i.e., rationale and answer. We find that CoT indeed improves MLLMs' adversarial robustness against the existing attack methods by leveraging the multi-step reasoning process, but not substantially. Based on our findings, we further propose a novel attack method, termed as stop-reasoning attack, that attacks the model while bypassing the CoT reasoning process. Experiments on three MLLMs and two visual reasoning datasets verify the effectiveness of our proposed method. We show that stop-reasoning attack can result in misled predictions and outperform baseline attacks by a significant margin.



## **25. Cloud Adversarial Example Generation for Remote Sensing Image Classification**

cs.CV

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.14240v1) [paper-pdf](http://arxiv.org/pdf/2409.14240v1)

**Authors**: Fei Ma, Yuqiang Feng, Fan Zhang, Yongsheng Zhou

**Abstract**: Most existing adversarial attack methods for remote sensing images merely add adversarial perturbations or patches, resulting in unnatural modifications. Clouds are common atmospheric effects in remote sensing images. Generating clouds on these images can produce adversarial examples better aligning with human perception. In this paper, we propose a Perlin noise based cloud generation attack method. Common Perlin noise based cloud generation is a random, non-optimizable process, which cannot be directly used to attack the target models. We design a Perlin Gradient Generator Network (PGGN), which takes a gradient parameter vector as input and outputs the grids of Perlin noise gradient vectors at different scales. After a series of computations based on the gradient vectors, cloud masks at corresponding scales can be produced. These cloud masks are then weighted and summed depending on a mixing coefficient vector and a scaling factor to produce the final cloud masks. The gradient vector, coefficient vector and scaling factor are collectively represented as a cloud parameter vector, transforming the cloud generation into a black-box optimization problem. The Differential Evolution (DE) algorithm is employed to solve for the optimal solution of the cloud parameter vector, achieving a query-based black-box attack. Detailed experiments confirm that this method has strong attack capabilities and achieves high query efficiency. Additionally, we analyze the transferability of the generated adversarial examples and their robustness in adversarial defense scenarios.



## **26. Adversarial Attacks on Parts of Speech: An Empirical Study in Text-to-Image Generation**

cs.CL

Findings of the EMNLP 2024

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2409.15381v1) [paper-pdf](http://arxiv.org/pdf/2409.15381v1)

**Authors**: G M Shahariar, Jia Chen, Jiachen Li, Yue Dong

**Abstract**: Recent studies show that text-to-image (T2I) models are vulnerable to adversarial attacks, especially with noun perturbations in text prompts. In this study, we investigate the impact of adversarial attacks on different POS tags within text prompts on the images generated by T2I models. We create a high-quality dataset for realistic POS tag token swapping and perform gradient-based attacks to find adversarial suffixes that mislead T2I models into generating images with altered tokens. Our empirical results show that the attack success rate (ASR) varies significantly among different POS tag categories, with nouns, proper nouns, and adjectives being the easiest to attack. We explore the mechanism behind the steering effect of adversarial suffixes, finding that the number of critical tokens and content fusion vary among POS tags, while features like suffix transferability are consistent across categories. We have made our implementation publicly available at - https://github.com/shahariar-shibli/Adversarial-Attack-on-POS-Tags.



## **27. RAMP: Boosting Adversarial Robustness Against Multiple $l_p$ Perturbations for Universal Robustness**

cs.LG

**SubmitDate**: 2024-09-21    [abs](http://arxiv.org/abs/2402.06827v2) [paper-pdf](http://arxiv.org/pdf/2402.06827v2)

**Authors**: Enyi Jiang, Gagandeep Singh

**Abstract**: Most existing works focus on improving robustness against adversarial attacks bounded by a single $l_p$ norm using adversarial training (AT). However, these AT models' multiple-norm robustness (union accuracy) is still low, which is crucial since in the real-world an adversary is not necessarily bounded by a single norm. The tradeoffs among robustness against multiple $l_p$ perturbations and accuracy/robustness make obtaining good union and clean accuracy challenging. We design a logit pairing loss to improve the union accuracy by analyzing the tradeoffs from the lens of distribution shifts. We connect natural training (NT) with AT via gradient projection, to incorporate useful information from NT into AT, where we empirically and theoretically show it moderates the accuracy/robustness tradeoff. We propose a novel training framework \textbf{RAMP}, to boost the robustness against multiple $l_p$ perturbations. \textbf{RAMP} can be easily adapted for robust fine-tuning and full AT. For robust fine-tuning, \textbf{RAMP} obtains a union accuracy up to $53.3\%$ on CIFAR-10, and $29.1\%$ on ImageNet. For training from scratch, \textbf{RAMP} achieves a union accuracy of $44.6\%$ and good clean accuracy of $81.2\%$ on ResNet-18 against AutoAttack on CIFAR-10. Beyond multi-norm robustness \textbf{RAMP}-trained models achieve superior \textit{universal robustness}, effectively generalizing against a range of unseen adversaries and natural corruptions.



## **28. $DA^3$: A Distribution-Aware Adversarial Attack against Language Models**

cs.CL

First two authors contribute equally; The paper has been accepted to  EMNLP2024 main conference

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2311.08598v3) [paper-pdf](http://arxiv.org/pdf/2311.08598v3)

**Authors**: Yibo Wang, Xiangjue Dong, James Caverlee, Philip S. Yu

**Abstract**: Language models can be manipulated by adversarial attacks, which introduce subtle perturbations to input data. While recent attack methods can achieve a relatively high attack success rate (ASR), we've observed that the generated adversarial examples have a different data distribution compared with the original examples. Specifically, these adversarial examples exhibit reduced confidence levels and greater divergence from the training data distribution. Consequently, they are easy to detect using straightforward detection methods, diminishing the efficacy of such attacks. To address this issue, we propose a Distribution-Aware Adversarial Attack ($DA^3$) method. $DA^3$ considers the distribution shifts of adversarial examples to improve attacks' effectiveness under detection methods. We further design a novel evaluation metric, the Non-detectable Attack Success Rate (NASR), which integrates both ASR and detectability for the attack task. We conduct experiments on four widely used datasets to validate the attack effectiveness and transferability of adversarial examples generated by $DA^3$ against both the white-box BERT-base and RoBERTa-base models and the black-box LLaMA2-7b model.



## **29. Persistent Backdoor Attacks in Continual Learning**

cs.LG

18 pages, 15 figures, 6 tables

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13864v1) [paper-pdf](http://arxiv.org/pdf/2409.13864v1)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.



## **30. ViTGuard: Attention-aware Detection against Adversarial Examples for Vision Transformer**

cs.CV

To appear in the Annual Computer Security Applications Conference  (ACSAC) 2024

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13828v1) [paper-pdf](http://arxiv.org/pdf/2409.13828v1)

**Authors**: Shihua Sun, Kenechukwu Nwodo, Shridatt Sugrim, Angelos Stavrou, Haining Wang

**Abstract**: The use of transformers for vision tasks has challenged the traditional dominant role of convolutional neural networks (CNN) in computer vision (CV). For image classification tasks, Vision Transformer (ViT) effectively establishes spatial relationships between patches within images, directing attention to important areas for accurate predictions. However, similar to CNNs, ViTs are vulnerable to adversarial attacks, which mislead the image classifier into making incorrect decisions on images with carefully designed perturbations. Moreover, adversarial patch attacks, which introduce arbitrary perturbations within a small area, pose a more serious threat to ViTs. Even worse, traditional detection methods, originally designed for CNN models, are impractical or suffer significant performance degradation when applied to ViTs, and they generally overlook patch attacks.   In this paper, we propose ViTGuard as a general detection method for defending ViT models against adversarial attacks, including typical attacks where perturbations spread over the entire input and patch attacks. ViTGuard uses a Masked Autoencoder (MAE) model to recover randomly masked patches from the unmasked regions, providing a flexible image reconstruction strategy. Then, threshold-based detectors leverage distinctive ViT features, including attention maps and classification (CLS) token representations, to distinguish between normal and adversarial samples. The MAE model does not involve any adversarial samples during training, ensuring the effectiveness of our detectors against unseen attacks. ViTGuard is compared with seven existing detection methods under nine attacks across three datasets. The evaluation results show the superiority of ViTGuard over existing detectors. Finally, considering the potential detection evasion, we further demonstrate ViTGuard's robustness against adaptive attacks for evasion.



## **31. Neurosymbolic Conformal Classification**

cs.LG

10 pages, 0 figures. arXiv admin note: text overlap with  arXiv:2404.08404

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13585v1) [paper-pdf](http://arxiv.org/pdf/2409.13585v1)

**Authors**: Arthur Ledaguenel, Céline Hudelot, Mostepha Khouadjia

**Abstract**: The last decades have seen a drastic improvement of Machine Learning (ML), mainly driven by Deep Learning (DL). However, despite the resounding successes of ML in many domains, the impossibility to provide guarantees of conformity and the fragility of ML systems (faced with distribution shifts, adversarial attacks, etc.) have prevented the design of trustworthy AI systems. Several research paths have been investigated to mitigate this fragility and provide some guarantees regarding the behavior of ML systems, among which are neurosymbolic AI and conformal prediction. Neurosymbolic artificial intelligence is a growing field of research aiming to combine neural network learning capabilities with the reasoning abilities of symbolic systems. One of the objective of this hybridization can be to provide theoritical guarantees that the output of the system will comply with some prior knowledge. Conformal prediction is a set of techniques that enable to take into account the uncertainty of ML systems by transforming the unique prediction into a set of predictions, called a confidence set. Interestingly, this comes with statistical guarantees regarding the presence of the true label inside the confidence set. Both approaches are distribution-free and model-agnostic. In this paper, we see how these two approaches can complement one another. We introduce several neurosymbolic conformal prediction techniques and explore their different characteristics (size of confidence sets, computational complexity, etc.).



## **32. Efficient Visualization of Neural Networks with Generative Models and Adversarial Perturbations**

cs.CV

4 pages, 3 figures

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13559v1) [paper-pdf](http://arxiv.org/pdf/2409.13559v1)

**Authors**: Athanasios Karagounis

**Abstract**: This paper presents a novel approach for deep visualization via a generative network, offering an improvement over existing methods. Our model simplifies the architecture by reducing the number of networks used, requiring only a generator and a discriminator, as opposed to the multiple networks traditionally involved. Additionally, our model requires less prior training knowledge and uses a non-adversarial training process, where the discriminator acts as a guide rather than a competitor to the generator. The core contribution of this work is its ability to generate detailed visualization images that align with specific class labels. Our model incorporates a unique skip-connection-inspired block design, which enhances label-directed image generation by propagating class information across multiple layers. Furthermore, we explore how these generated visualizations can be utilized as adversarial examples, effectively fooling classification networks with minimal perceptible modifications to the original images. Experimental results demonstrate that our method outperforms traditional adversarial example generation techniques in both targeted and non-targeted attacks, achieving up to a 94.5% fooling rate with minimal perturbation. This work bridges the gap between visualization methods and adversarial examples, proposing that fooling rate could serve as a quantitative measure for evaluating visualization quality. The insights from this study provide a new perspective on the interpretability of neural networks and their vulnerabilities to adversarial attacks.



## **33. Deterministic versus stochastic dynamical classifiers: opposing random adversarial attacks with noise**

cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13470v1) [paper-pdf](http://arxiv.org/pdf/2409.13470v1)

**Authors**: Lorenzo Chicchi, Duccio Fanelli, Diego Febbe, Lorenzo Buffoni, Francesca Di Patti, Lorenzo Giambagli, Raffele Marino

**Abstract**: The Continuous-Variable Firing Rate (CVFR) model, widely used in neuroscience to describe the intertangled dynamics of excitatory biological neurons, is here trained and tested as a veritable dynamically assisted classifier. To this end the model is supplied with a set of planted attractors which are self-consistently embedded in the inter-nodes coupling matrix, via its spectral decomposition. Learning to classify amounts to sculp the basin of attraction of the imposed equilibria, directing different items towards the corresponding destination target, which reflects the class of respective pertinence. A stochastic variant of the CVFR model is also studied and found to be robust to aversarial random attacks, which corrupt the items to be classified. This remarkable finding is one of the very many surprising effects which arise when noise and dynamical attributes are made to mutually resonate.



## **34. Celtibero: Robust Layered Aggregation for Federated Learning**

cs.CR

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.14240v2) [paper-pdf](http://arxiv.org/pdf/2408.14240v2)

**Authors**: Borja Molina-Coronado

**Abstract**: Federated Learning (FL) is an innovative approach to distributed machine learning. While FL offers significant privacy advantages, it also faces security challenges, particularly from poisoning attacks where adversaries deliberately manipulate local model updates to degrade model performance or introduce hidden backdoors. Existing defenses against these attacks have been shown to be effective when the data on the nodes is identically and independently distributed (i.i.d.), but they often fail under less restrictive, non-i.i.d data conditions. To overcome these limitations, we introduce Celtibero, a novel defense mechanism that integrates layered aggregation to enhance robustness against adversarial manipulation. Through extensive experiments on the MNIST and IMDB datasets, we demonstrate that Celtibero consistently achieves high main task accuracy (MTA) while maintaining minimal attack success rates (ASR) across a range of untargeted and targeted poisoning attacks. Our results highlight the superiority of Celtibero over existing defenses such as FL-Defender, LFighter, and FLAME, establishing it as a highly effective solution for securing federated learning systems against sophisticated poisoning attacks.



## **35. Enhancing Transferability of Adversarial Attacks with GE-AdvGAN+: A Comprehensive Framework for Gradient Editing**

cs.AI

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2408.12673v3) [paper-pdf](http://arxiv.org/pdf/2408.12673v3)

**Authors**: Zhibo Jin, Jiayu Zhang, Zhiyu Zhu, Chenyu Zhang, Jiahao Huang, Jianlong Zhou, Fang Chen

**Abstract**: Transferable adversarial attacks pose significant threats to deep neural networks, particularly in black-box scenarios where internal model information is inaccessible. Studying adversarial attack methods helps advance the performance of defense mechanisms and explore model vulnerabilities. These methods can uncover and exploit weaknesses in models, promoting the development of more robust architectures. However, current methods for transferable attacks often come with substantial computational costs, limiting their deployment and application, especially in edge computing scenarios. Adversarial generative models, such as Generative Adversarial Networks (GANs), are characterized by their ability to generate samples without the need for retraining after an initial training phase. GE-AdvGAN, a recent method for transferable adversarial attacks, is based on this principle. In this paper, we propose a novel general framework for gradient editing-based transferable attacks, named GE-AdvGAN+, which integrates nearly all mainstream attack methods to enhance transferability while significantly reducing computational resource consumption. Our experiments demonstrate the compatibility and effectiveness of our framework. Compared to the baseline AdvGAN, our best-performing method, GE-AdvGAN++, achieves an average ASR improvement of 47.8. Additionally, it surpasses the latest competing algorithm, GE-AdvGAN, with an average ASR increase of 5.9. The framework also exhibits enhanced computational efficiency, achieving 2217.7 FPS, outperforming traditional methods such as BIM and MI-FGSM. The implementation code for our GE-AdvGAN+ framework is available at https://github.com/GEAdvGANP



## **36. Relationship between Uncertainty in DNNs and Adversarial Attacks**

cs.LG

review

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13232v1) [paper-pdf](http://arxiv.org/pdf/2409.13232v1)

**Authors**: Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.



## **37. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13174v1) [paper-pdf](http://arxiv.org/pdf/2409.13174v1)

**Authors**: Hao Cheng, Erjia Xiao, Chengyuan Yu, Zhao Yao, Jiahang Cao, Qiang Zhang, Jiaxu Wang, Mengshu Sun, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical security threats.



## **38. Hidden Activations Are Not Enough: A General Approach to Neural Network Predictions**

cs.LG

**SubmitDate**: 2024-09-20    [abs](http://arxiv.org/abs/2409.13163v1) [paper-pdf](http://arxiv.org/pdf/2409.13163v1)

**Authors**: Samuel Leblanc, Aiky Rasolomanana, Marco Armenta

**Abstract**: We introduce a novel mathematical framework for analyzing neural networks using tools from quiver representation theory. This framework enables us to quantify the similarity between a new data sample and the training data, as perceived by the neural network. By leveraging the induced quiver representation of a data sample, we capture more information than traditional hidden layer outputs. This quiver representation abstracts away the complexity of the computations of the forward pass into a single matrix, allowing us to employ simple geometric and statistical arguments in a matrix space to study neural network predictions. Our mathematical results are architecture-agnostic and task-agnostic, making them broadly applicable. As proof of concept experiments, we apply our results for the MNIST and FashionMNIST datasets on the problem of detecting adversarial examples on different MLP architectures and several adversarial attack methods. Our experiments can be reproduced with our \href{https://github.com/MarcoArmenta/Hidden-Activations-are-not-Enough}{publicly available repository}.



## **39. FedAT: Federated Adversarial Training for Distributed Insider Threat Detection**

cs.CR

10 pages, 7 figures

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.13083v1) [paper-pdf](http://arxiv.org/pdf/2409.13083v1)

**Authors**: R G Gayathri, Atul Sajjanhar, Md Palash Uddin, Yong Xiang

**Abstract**: Insider threats usually occur from within the workplace, where the attacker is an entity closely associated with the organization. The sequence of actions the entities take on the resources to which they have access rights allows us to identify the insiders. Insider Threat Detection (ITD) using Machine Learning (ML)-based approaches gained attention in the last few years. However, most techniques employed centralized ML methods to perform such an ITD. Organizations operating from multiple locations cannot contribute to the centralized models as the data is generated from various locations. In particular, the user behavior data, which is the primary source of ITD, cannot be shared among the locations due to privacy concerns. Additionally, the data distributed across various locations result in extreme class imbalance due to the rarity of attacks. Federated Learning (FL), a distributed data modeling paradigm, gained much interest recently. However, FL-enabled ITD is not yet explored, and it still needs research to study the significant issues of its implementation in practical settings. As such, our work investigates an FL-enabled multiclass ITD paradigm that considers non-Independent and Identically Distributed (non-IID) data distribution to detect insider threats from different locations (clients) of an organization. Specifically, we propose a Federated Adversarial Training (FedAT) approach using a generative model to alleviate the extreme data skewness arising from the non-IID data distribution among the clients. Besides, we propose to utilize a Self-normalized Neural Network-based Multi-Layer Perceptron (SNN-MLP) model to improve ITD. We perform comprehensive experiments and compare the results with the benchmarks to manifest the enhanced performance of the proposed FedATdriven ITD scheme.



## **40. Defending against Reverse Preference Attacks is Difficult**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12914v1) [paper-pdf](http://arxiv.org/pdf/2409.12914v1)

**Authors**: Domenic Rosati, Giles Edkins, Harsh Raj, David Atanasov, Subhabrata Majumdar, Janarthanan Rajendran, Frank Rudzicz, Hassan Sajjad

**Abstract**: While there has been progress towards aligning Large Language Models (LLMs) with human values and ensuring safe behaviour at inference time, safety-aligned LLMs are known to be vulnerable to training-time attacks such as supervised fine-tuning (SFT) on harmful datasets. In this paper, we ask if LLMs are vulnerable to adversarial reinforcement learning. Motivated by this goal, we propose Reverse Preference Attacks (RPA), a class of attacks to make LLMs learn harmful behavior using adversarial reward during reinforcement learning from human feedback (RLHF). RPAs expose a critical safety gap of safety-aligned LLMs in RL settings: they easily explore the harmful text generation policies to optimize adversarial reward. To protect against RPAs, we explore a host of mitigation strategies. Leveraging Constrained Markov-Decision Processes, we adapt a number of mechanisms to defend against harmful fine-tuning attacks into the RL setting. Our experiments show that ``online" defenses that are based on the idea of minimizing the negative log likelihood of refusals -- with the defender having control of the loss function -- can effectively protect LLMs against RPAs. However, trying to defend model weights using ``offline" defenses that operate under the assumption that the defender has no control over the loss function are less effective in the face of RPAs. These findings show that attacks done using RL can be used to successfully undo safety alignment in open-weight LLMs and use them for malicious purposes.



## **41. VCAT: Vulnerability-aware and Curiosity-driven Adversarial Training for Enhancing Autonomous Vehicle Robustness**

cs.LG

7 pages, 5 figures, conference

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12997v1) [paper-pdf](http://arxiv.org/pdf/2409.12997v1)

**Authors**: Xuan Cai, Zhiyong Cui, Xuesong Bai, Ruimin Ke, Zhenshu Ma, Haiyang Yu, Yilong Ren

**Abstract**: Autonomous vehicles (AVs) face significant threats to their safe operation in complex traffic environments. Adversarial training has emerged as an effective method of enabling AVs to preemptively fortify their robustness against malicious attacks. Train an attacker using an adversarial policy, allowing the AV to learn robust driving through interaction with this attacker. However, adversarial policies in existing methodologies often get stuck in a loop of overexploiting established vulnerabilities, resulting in poor improvement for AVs. To overcome the limitations, we introduce a pioneering framework termed Vulnerability-aware and Curiosity-driven Adversarial Training (VCAT). Specifically, during the traffic vehicle attacker training phase, a surrogate network is employed to fit the value function of the AV victim, providing dense information about the victim's inherent vulnerabilities. Subsequently, random network distillation is used to characterize the novelty of the environment, constructing an intrinsic reward to guide the attacker in exploring unexplored territories. In the victim defense training phase, the AV is trained in critical scenarios in which the pretrained attacker is positioned around the victim to generate attack behaviors. Experimental results revealed that the training methodology provided by VCAT significantly improved the robust control capabilities of learning-based AVs, outperforming both conventional training modalities and alternative reinforcement learning counterparts, with a marked reduction in crash rates. The code is available at https://github.com/caixxuan/VCAT.



## **42. Boosting Certified Robustness for Time Series Classification with Efficient Self-Ensemble**

cs.LG

6 figures, 4 tables, 10 pages

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.02802v3) [paper-pdf](http://arxiv.org/pdf/2409.02802v3)

**Authors**: Chang Dong, Zhengyang Li, Liangwei Zheng, Weitong Chen, Wei Emma Zhang

**Abstract**: Recently, the issue of adversarial robustness in the time series domain has garnered significant attention. However, the available defense mechanisms remain limited, with adversarial training being the predominant approach, though it does not provide theoretical guarantees. Randomized Smoothing has emerged as a standout method due to its ability to certify a provable lower bound on robustness radius under $\ell_p$-ball attacks. Recognizing its success, research in the time series domain has started focusing on these aspects. However, existing research predominantly focuses on time series forecasting, or under the non-$\ell_p$ robustness in statistic feature augmentation for time series classification~(TSC). Our review found that Randomized Smoothing performs modestly in TSC, struggling to provide effective assurances on datasets with poor robustness. Therefore, we propose a self-ensemble method to enhance the lower bound of the probability confidence of predicted labels by reducing the variance of classification margins, thereby certifying a larger radius. This approach also addresses the computational overhead issue of Deep Ensemble~(DE) while remaining competitive and, in some cases, outperforming it in terms of robustness. Both theoretical analysis and experimental results validate the effectiveness of our method, demonstrating superior performance in robustness testing compared to baseline approaches.



## **43. Deep generative models as an adversarial attack strategy for tabular machine learning**

cs.LG

Accepted at ICMLC 2024 (International Conference on Machine Learning  and Cybernetics)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12642v1) [paper-pdf](http://arxiv.org/pdf/2409.12642v1)

**Authors**: Salijona Dyrmishi, Mihaela Cătălina Stoian, Eleonora Giunchiglia, Maxime Cordy

**Abstract**: Deep Generative Models (DGMs) have found application in computer vision for generating adversarial examples to test the robustness of machine learning (ML) systems. Extending these adversarial techniques to tabular ML presents unique challenges due to the distinct nature of tabular data and the necessity to preserve domain constraints in adversarial examples. In this paper, we adapt four popular tabular DGMs into adversarial DGMs (AdvDGMs) and evaluate their effectiveness in generating realistic adversarial examples that conform to domain constraints.



## **44. Exploring Privacy and Fairness Risks in Sharing Diffusion Models: An Adversarial Perspective**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2402.18607v3) [paper-pdf](http://arxiv.org/pdf/2402.18607v3)

**Authors**: Xinjian Luo, Yangfan Jiang, Fei Wei, Yuncheng Wu, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Diffusion models have recently gained significant attention in both academia and industry due to their impressive generative performance in terms of both sampling quality and distribution coverage. Accordingly, proposals are made for sharing pre-trained diffusion models across different organizations, as a way of improving data utilization while enhancing privacy protection by avoiding sharing private data directly. However, the potential risks associated with such an approach have not been comprehensively examined.   In this paper, we take an adversarial perspective to investigate the potential privacy and fairness risks associated with the sharing of diffusion models. Specifically, we investigate the circumstances in which one party (the sharer) trains a diffusion model using private data and provides another party (the receiver) black-box access to the pre-trained model for downstream tasks. We demonstrate that the sharer can execute fairness poisoning attacks to undermine the receiver's downstream models by manipulating the training data distribution of the diffusion model. Meanwhile, the receiver can perform property inference attacks to reveal the distribution of sensitive features in the sharer's dataset. Our experiments conducted on real-world datasets demonstrate remarkable attack performance on different types of diffusion models, which highlights the critical importance of robust data auditing and privacy protection protocols in pertinent applications.



## **45. Adversarial Attack for Explanation Robustness of Rationalization Models**

cs.CL

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2408.10795v3) [paper-pdf](http://arxiv.org/pdf/2408.10795v3)

**Authors**: Yuankai Zhang, Lingxiao Kong, Haozhao Wang, Ruixuan Li, Jun Wang, Yuhua Li, Wei Liu

**Abstract**: Rationalization models, which select a subset of input text as rationale-crucial for humans to understand and trust predictions-have recently emerged as a prominent research area in eXplainable Artificial Intelligence. However, most of previous studies mainly focus on improving the quality of the rationale, ignoring its robustness to malicious attack. Specifically, whether the rationalization models can still generate high-quality rationale under the adversarial attack remains unknown. To explore this, this paper proposes UAT2E, which aims to undermine the explainability of rationalization models without altering their predictions, thereby eliciting distrust in these models from human users. UAT2E employs the gradient-based search on triggers and then inserts them into the original input to conduct both the non-target and target attack. Experimental results on five datasets reveal the vulnerability of rationalization models in terms of explanation, where they tend to select more meaningless tokens under attacks. Based on this, we make a series of recommendations for improving rationalization models in terms of explanation.



## **46. Designing an attack-defense game: how to increase robustness of financial transaction models via a competition**

cs.LG

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2308.11406v3) [paper-pdf](http://arxiv.org/pdf/2308.11406v3)

**Authors**: Alexey Zaytsev, Maria Kovaleva, Alex Natekin, Evgeni Vorsin, Valerii Smirnov, Georgii Smirnov, Oleg Sidorshin, Alexander Senin, Alexander Dudin, Dmitry Berestnev

**Abstract**: Banks routinely use neural networks to make decisions. While these models offer higher accuracy, they are susceptible to adversarial attacks, a risk often overlooked in the context of event sequences, particularly sequences of financial transactions, as most works consider computer vision and NLP modalities.   We propose a thorough approach to studying these risks: a novel type of competition that allows a realistic and detailed investigation of problems in financial transaction data. The participants directly oppose each other, proposing attacks and defenses -- so they are examined in close-to-real-life conditions.   The paper outlines our unique competition structure with direct opposition of participants, presents results for several different top submissions, and analyzes the competition results. We also introduce a new open dataset featuring financial transactions with credit default labels, enhancing the scope for practical research and development.



## **47. Magmaw: Modality-Agnostic Adversarial Attacks on Machine Learning-Based Wireless Communication Systems**

cs.CR

Accepted at NDSS 2025

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2311.00207v2) [paper-pdf](http://arxiv.org/pdf/2311.00207v2)

**Authors**: Jung-Woo Chang, Ke Sun, Nasimeh Heydaribeni, Seira Hidano, Xinyu Zhang, Farinaz Koushanfar

**Abstract**: Machine Learning (ML) has been instrumental in enabling joint transceiver optimization by merging all physical layer blocks of the end-to-end wireless communication systems. Although there have been a number of adversarial attacks on ML-based wireless systems, the existing methods do not provide a comprehensive view including multi-modality of the source data, common physical layer protocols, and wireless domain constraints. This paper proposes Magmaw, a novel wireless attack methodology capable of generating universal adversarial perturbations for any multimodal signal transmitted over a wireless channel. We further introduce new objectives for adversarial attacks on downstream applications. We adopt the widely-used defenses to verify the resilience of Magmaw. For proof-of-concept evaluation, we build a real-time wireless attack platform using a software-defined radio system. Experimental results demonstrate that Magmaw causes significant performance degradation even in the presence of strong defense mechanisms. Furthermore, we validate the performance of Magmaw in two case studies: encrypted communication channel and channel modality-based ML model.



## **48. TEAM: Temporal Adversarial Examples Attack Model against Network Intrusion Detection System Applied to RNN**

cs.CR

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.12472v1) [paper-pdf](http://arxiv.org/pdf/2409.12472v1)

**Authors**: Ziyi Liu, Dengpan Ye, Long Tang, Yunming Zhang, Jiacheng Deng

**Abstract**: With the development of artificial intelligence, neural networks play a key role in network intrusion detection systems (NIDS). Despite the tremendous advantages, neural networks are susceptible to adversarial attacks. To improve the reliability of NIDS, many research has been conducted and plenty of solutions have been proposed. However, the existing solutions rarely consider the adversarial attacks against recurrent neural networks (RNN) with time steps, which would greatly affect the application of NIDS in real world. Therefore, we first propose a novel RNN adversarial attack model based on feature reconstruction called \textbf{T}emporal adversarial \textbf{E}xamples \textbf{A}ttack \textbf{M}odel \textbf{(TEAM)}, which applied to time series data and reveals the potential connection between adversarial and time steps in RNN. That is, the past adversarial examples within the same time steps can trigger further attacks on current or future original examples. Moreover, TEAM leverages Time Dilation (TD) to effectively mitigates the effect of temporal among adversarial examples within the same time steps. Experimental results show that in most attack categories, TEAM improves the misjudgment rate of NIDS on both black and white boxes, making the misjudgment rate reach more than 96.68%. Meanwhile, the maximum increase in the misjudgment rate of the NIDS for subsequent original samples exceeds 95.57%.



## **49. Object-fabrication Targeted Attack for Object Detection**

cs.CV

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2212.06431v3) [paper-pdf](http://arxiv.org/pdf/2212.06431v3)

**Authors**: Xuchong Zhang, Changfeng Sun, Haoliang Han, Hongbin Sun

**Abstract**: Recent studies have demonstrated that object detection networks are usually vulnerable to adversarial examples. Generally, adversarial attacks for object detection can be categorized into targeted and untargeted attacks. Compared with untargeted attacks, targeted attacks present greater challenges and all existing targeted attack methods launch the attack by misleading detectors to mislabel the detected object as a specific wrong label. However, since these methods must depend on the presence of the detected objects within the victim image, they suffer from limitations in attack scenarios and attack success rates. In this paper, we propose a targeted feature space attack method that can mislead detectors to `fabricate' extra designated objects regardless of whether the victim image contains objects or not. Specifically, we introduce a guided image to extract coarse-grained features of the target objects and design an innovative dual attention mechanism to filter out the critical features of the target objects efficiently. The attack performance of the proposed method is evaluated on MS COCO and BDD100K datasets with FasterRCNN and YOLOv5. Evaluation results indicate that the proposed targeted feature space attack method shows significant improvements in terms of image-specific, universality, and generalization attack performance, compared with the previous targeted attack for object detection.



## **50. Towards Physically-Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

8 pages, 6 figures, submitted to the 2025 IEEE International  Conference on Robotics & Automation (ICRA)

**SubmitDate**: 2024-09-19    [abs](http://arxiv.org/abs/2409.10071v2) [paper-pdf](http://arxiv.org/pdf/2409.10071v2)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The deployment of embodied navigation agents in safety-critical environments raises concerns about their vulnerability to adversarial attacks on deep neural networks. However, current attack methods often lack practicality due to challenges in transitioning from the digital to the physical world, while existing physical attacks for object detection fail to achieve both multi-view effectiveness and naturalness. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches with learnable textures and opacity to objects. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which uses feedback from the navigation model to optimize the patch's texture. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, where opacity is refined after texture optimization. Experimental results show our adversarial patches reduce navigation success rates by about 40%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: [https://github.com/chen37058/Physical-Attacks-in-Embodied-Navigation].



