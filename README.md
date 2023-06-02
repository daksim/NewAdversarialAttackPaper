# Latest Adversarial Attack Papers
**update at 2023-06-02 11:57:32**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Intriguing Properties of Text-guided Diffusion Models**

cs.CV

Code will be available at: https://github.com/qihao067/SAGE/

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00974v1) [paper-pdf](http://arxiv.org/pdf/2306.00974v1)

**Authors**: Qihao Liu, Adam Kortylewski, Yutong Bai, Song Bai, Alan Yuille

**Abstract**: Text-guided diffusion models (TDMs) are widely applied but can fail unexpectedly. Common failures include: (i) natural-looking text prompts generating images with the wrong content, or (ii) different random samples of the latent variables that generate vastly different, and even unrelated, outputs despite being conditioned on the same text prompt. In this work, we aim to study and understand the failure modes of TDMs in more detail. To achieve this, we propose SAGE, an adversarial attack on TDMs that uses image classifiers as surrogate loss functions, to search over the discrete prompt space and the high-dimensional latent space of TDMs to automatically discover unexpected behaviors and failure cases in the image generation. We make several technical contributions to ensure that SAGE finds failure cases of the diffusion model, rather than the classifier, and verify this in a human study. Our study reveals four intriguing properties of TDMs that have not been systematically studied before: (1) We find a variety of natural text prompts producing images that fail to capture the semantics of input texts. We categorize these failures into ten distinct types based on the underlying causes. (2) We find samples in the latent space (which are not outliers) that lead to distorted images independent of the text prompt, suggesting that parts of the latent space are not well-structured. (3) We also find latent samples that lead to natural-looking images which are unrelated to the text prompt, implying a potential misalignment between the latent and prompt spaces. (4) By appending a single adversarial token embedding to an input prompt we can generate a variety of specified target objects, while only minimally affecting the CLIP score. This demonstrates the fragility of language representations and raises potential safety concerns.



## **2. Adversarial Robustness in Unsupervised Machine Learning: A Systematic Review**

cs.LG

38 pages, 11 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00687v1) [paper-pdf](http://arxiv.org/pdf/2306.00687v1)

**Authors**: Mathias Lundteigen Mohus, Jinyue Li

**Abstract**: As the adoption of machine learning models increases, ensuring robust models against adversarial attacks is increasingly important. With unsupervised machine learning gaining more attention, ensuring it is robust against attacks is vital. This paper conducts a systematic literature review on the robustness of unsupervised learning, collecting 86 papers. Our results show that most research focuses on privacy attacks, which have effective defenses; however, many attacks lack effective and general defensive measures. Based on the results, we formulate a model on the properties of an attack on unsupervised learning, contributing to future research by providing a model to use.



## **3. Byzantine-Robust Clustered Federated Learning**

stat.ML

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00638v1) [paper-pdf](http://arxiv.org/pdf/2306.00638v1)

**Authors**: Zhixu Tao, Kun Yang, Sanjeev R. Kulkarni

**Abstract**: This paper focuses on the problem of adversarial attacks from Byzantine machines in a Federated Learning setting where non-Byzantine machines can be partitioned into disjoint clusters. In this setting, non-Byzantine machines in the same cluster have the same underlying data distribution, and different clusters of non-Byzantine machines have different learning tasks. Byzantine machines can adversarially attack any cluster and disturb the training process on clusters they attack. In the presence of Byzantine machines, the goal of our work is to identify cluster membership of non-Byzantine machines and optimize the models learned by each cluster. We adopt the Iterative Federated Clustering Algorithm (IFCA) framework of Ghosh et al. (2020) to alternatively estimate cluster membership and optimize models. In order to make this framework robust against adversarial attacks from Byzantine machines, we use coordinate-wise trimmed mean and coordinate-wise median aggregation methods used by Yin et al. (2018). Specifically, we propose a new Byzantine-Robust Iterative Federated Clustering Algorithm to improve on the results in Ghosh et al. (2019). We prove a convergence rate for this algorithm for strongly convex loss functions. We compare our convergence rate with the convergence rate of an existing algorithm, and we demonstrate the performance of our algorithm on simulated data.



## **4. Spying on the Spy: Security Analysis of Hidden Cameras**

cs.CR

19 pages. Conference: NSS 2023: 17th International Conference on  Network and System Security

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00610v1) [paper-pdf](http://arxiv.org/pdf/2306.00610v1)

**Authors**: Samuel Herodotou, Feng Hao

**Abstract**: Hidden cameras, also called spy cameras, are surveillance tools commonly used to spy on people without their knowledge. Whilst previous studies largely focused on investigating the detection of such a camera and the privacy implications, the security of the camera itself has received limited attention. Compared with ordinary IP cameras, spy cameras are normally sold in bulk at cheap prices and are ubiquitously deployed in hidden places within homes and workplaces. A security compromise of these cameras can have severe consequences. In this paper, we analyse a generic IP camera module, which has been packaged and re-branded for sale by several spy camera vendors. The module is controlled by mobile phone apps. By analysing the Android app and the traffic data, we reverse-engineered the security design of the whole system, including the module's Linux OS environment, the file structure, the authentication mechanism, the session management, and the communication with a remote server. Serious vulnerabilities have been identified in every component. Combined together, they allow an adversary to take complete control of a spy camera from anywhere over the Internet, enabling arbitrary code execution. This is possible even if the camera is behind a firewall. All that an adversary needs to launch an attack is the camera's serial number, which users sometimes unknowingly share in online reviews. We responsibly disclosed our findings to the manufacturer. Whilst the manufacturer acknowledged our work, they showed no intention to fix the problems. Patching or recalling the affected cameras is infeasible due to complexities in the supply chain. However, it is prudent to assume that bad actors have already been exploiting these flaws. We provide details of the identified vulnerabilities in order to raise public awareness, especially on the grave danger of disclosing a spy camera's serial number.



## **5. Does Black-box Attribute Inference Attacks on Graph Neural Networks Constitute Privacy Risk?**

cs.LG

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00578v1) [paper-pdf](http://arxiv.org/pdf/2306.00578v1)

**Authors**: Iyiola E. Olatunji, Anmar Hizber, Oliver Sihlovec, Megha Khosla

**Abstract**: Graph neural networks (GNNs) have shown promising results on real-life datasets and applications, including healthcare, finance, and education. However, recent studies have shown that GNNs are highly vulnerable to attacks such as membership inference attack and link reconstruction attack. Surprisingly, attribute inference attacks has received little attention. In this paper, we initiate the first investigation into attribute inference attack where an attacker aims to infer the sensitive user attributes based on her public or non-sensitive attributes. We ask the question whether black-box attribute inference attack constitutes a significant privacy risk for graph-structured data and their corresponding GNN model. We take a systematic approach to launch the attacks by varying the adversarial knowledge and assumptions. Our findings reveal that when an attacker has black-box access to the target model, GNNs generally do not reveal significantly more information compared to missing value estimation techniques. Code is available.



## **6. Constructing Semantics-Aware Adversarial Examples with Probabilistic Perspective**

stat.ML

17 pages, 14 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00353v1) [paper-pdf](http://arxiv.org/pdf/2306.00353v1)

**Authors**: Andi Zhang, Damon Wischik

**Abstract**: In this study, we introduce a novel, probabilistic viewpoint on adversarial examples, achieved through box-constrained Langevin Monte Carlo (LMC). Proceeding from this perspective, we develop an innovative approach for generating semantics-aware adversarial examples in a principled manner. This methodology transcends the restriction imposed by geometric distance, instead opting for semantic constraints. Our approach empowers individuals to incorporate their personal comprehension of semantics into the model. Through human evaluation, we validate that our semantics-aware adversarial examples maintain their inherent meaning. Experimental findings on the MNIST and SVHN datasets demonstrate that our semantics-aware adversarial examples can effectively circumvent robust adversarial training methods tailored for traditional adversarial attacks.



## **7. CALICO: Self-Supervised Camera-LiDAR Contrastive Pre-training for BEV Perception**

cs.CV

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00349v1) [paper-pdf](http://arxiv.org/pdf/2306.00349v1)

**Authors**: Jiachen Sun, Haizhong Zheng, Qingzhao Zhang, Atul Prakash, Z. Morley Mao, Chaowei Xiao

**Abstract**: Perception is crucial in the realm of autonomous driving systems, where bird's eye view (BEV)-based architectures have recently reached state-of-the-art performance. The desirability of self-supervised representation learning stems from the expensive and laborious process of annotating 2D and 3D data. Although previous research has investigated pretraining methods for both LiDAR and camera-based 3D object detection, a unified pretraining framework for multimodal BEV perception is missing. In this study, we introduce CALICO, a novel framework that applies contrastive objectives to both LiDAR and camera backbones. Specifically, CALICO incorporates two stages: point-region contrast (PRC) and region-aware distillation (RAD). PRC better balances the region- and scene-level representation learning on the LiDAR modality and offers significant performance improvement compared to existing methods. RAD effectively achieves contrastive distillation on our self-trained teacher model. CALICO's efficacy is substantiated by extensive evaluations on 3D object detection and BEV map segmentation tasks, where it delivers significant performance improvements. Notably, CALICO outperforms the baseline method by 10.5% and 8.6% on NDS and mAP. Moreover, CALICO boosts the robustness of multimodal 3D object detection against adversarial attacks and corruption. Additionally, our framework can be tailored to different backbones and heads, positioning it as a promising approach for multimodal BEV perception.



## **8. Improving Adversarial Robustness by Putting More Regularizations on Less Robust Samples**

stat.ML

Accepted in ICML 2023

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2206.03353v4) [paper-pdf](http://arxiv.org/pdf/2206.03353v4)

**Authors**: Dongyoon Yang, Insung Kong, Yongdai Kim

**Abstract**: Adversarial training, which is to enhance robustness against adversarial attacks, has received much attention because it is easy to generate human-imperceptible perturbations of data to deceive a given deep neural network. In this paper, we propose a new adversarial training algorithm that is theoretically well motivated and empirically superior to other existing algorithms. A novel feature of the proposed algorithm is to apply more regularization to data vulnerable to adversarial attacks than other existing regularization algorithms do. Theoretically, we show that our algorithm can be understood as an algorithm of minimizing the regularized empirical risk motivated from a newly derived upper bound of the robust risk. Numerical experiments illustrate that our proposed algorithm improves the generalization (accuracy on examples) and robustness (accuracy on adversarial attacks) simultaneously to achieve the state-of-the-art performance.



## **9. Adversarial-Aware Deep Learning System based on a Secondary Classical Machine Learning Verification Approach**

cs.CR

17 pages, 3 figures

**SubmitDate**: 2023-06-01    [abs](http://arxiv.org/abs/2306.00314v1) [paper-pdf](http://arxiv.org/pdf/2306.00314v1)

**Authors**: Mohammed Alkhowaiter, Hisham Kholidy, Mnassar Alyami, Abdulmajeed Alghamdi, Cliff Zou

**Abstract**: Deep learning models have been used in creating various effective image classification applications. However, they are vulnerable to adversarial attacks that seek to misguide the models into predicting incorrect classes. Our study of major adversarial attack models shows that they all specifically target and exploit the neural networking structures in their designs. This understanding makes us develop a hypothesis that most classical machine learning models, such as Random Forest (RF), are immune to adversarial attack models because they do not rely on neural network design at all. Our experimental study of classical machine learning models against popular adversarial attacks supports this hypothesis. Based on this hypothesis, we propose a new adversarial-aware deep learning system by using a classical machine learning model as the secondary verification system to complement the primary deep learning model in image classification. Although the secondary classical machine learning model has less accurate output, it is only used for verification purposes, which does not impact the output accuracy of the primary deep learning model, and at the same time, can effectively detect an adversarial attack when a clear mismatch occurs. Our experiments based on CIFAR-100 dataset show that our proposed approach outperforms current state-of-the-art adversarial defense systems.



## **10. Near Optimal Adversarial Attack on UCB Bandits**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2008.09312v5) [paper-pdf](http://arxiv.org/pdf/2008.09312v5)

**Authors**: Shiliang Zuo

**Abstract**: I study a stochastic multi-arm bandit problem where rewards are subject to adversarial corruption. I propose a novel attack strategy that manipulates a learner employing the UCB algorithm into pulling some non-optimal target arm $T - o(T)$ times with a cumulative cost that scales as $\widehat{O}(\sqrt{\log T})$, where $T$ is the number of rounds. I also prove the first lower bound on the cumulative attack cost. The lower bound matches the upper bound up to $O(\log \log T)$ factors, showing the proposed attack strategy to be near optimal.



## **11. Deception by Omission: Using Adversarial Missingness to Poison Causal Structure Learning**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.20043v1) [paper-pdf](http://arxiv.org/pdf/2305.20043v1)

**Authors**: Deniz Koyuncu, Alex Gittens, Bülent Yener, Moti Yung

**Abstract**: Inference of causal structures from observational data is a key component of causal machine learning; in practice, this data may be incompletely observed. Prior work has demonstrated that adversarial perturbations of completely observed training data may be used to force the learning of inaccurate causal structural models (SCMs). However, when the data can be audited for correctness (e.g., it is crytographically signed by its source), this adversarial mechanism is invalidated. This work introduces a novel attack methodology wherein the adversary deceptively omits a portion of the true training data to bias the learned causal structures in a desired manner. Theoretically sound attack mechanisms are derived for the case of arbitrary SCMs, and a sample-efficient learning-based heuristic is given for Gaussian SCMs. Experimental validation of these approaches on real and synthetic data sets demonstrates the effectiveness of adversarial missingness attacks at deceiving popular causal structure learning algorithms.



## **12. IB-RAR: Information Bottleneck as Regularizer for Adversarial Robustness**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2302.10896v2) [paper-pdf](http://arxiv.org/pdf/2302.10896v2)

**Authors**: Xiaoyun Xu, Guilherme Perin, Stjepan Picek

**Abstract**: In this paper, we propose a novel method, IB-RAR, which uses Information Bottleneck (IB) to strengthen adversarial robustness for both adversarial training and non-adversarial-trained methods. We first use the IB theory to build regularizers as learning objectives in the loss function. Then, we filter out unnecessary features of intermediate representation according to their mutual information (MI) with labels, as the network trained with IB provides easily distinguishable MI for its features. Experimental results show that our method can be naturally combined with adversarial training and provides consistently better accuracy on new adversarial examples. Our method improves the accuracy by an average of 3.07% against five adversarial attacks for the VGG16 network, trained with three adversarial training benchmarks and the CIFAR-10 dataset. In addition, our method also provides good robustness for undefended methods, such as training with cross-entropy loss only. Finally, in the absence of adversarial training, the VGG16 network trained using our method and the CIFAR-10 dataset reaches an accuracy of 35.86% against PGD examples, while using all layers reaches 25.61% accuracy.



## **13. Graph-based methods coupled with specific distributional distances for adversarial attack detection**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2306.00042v1) [paper-pdf](http://arxiv.org/pdf/2306.00042v1)

**Authors**: Dwight Nwaigwe, Lucrezia Carboni, Martial Mermillod, Sophie Achard, Michel Dojat

**Abstract**: Artificial neural networks are prone to being fooled by carefully perturbed inputs which cause an egregious misclassification. These \textit{adversarial} attacks have been the focus of extensive research. Likewise, there has been an abundance of research in ways to detect and defend against them. We introduce a novel approach of detection and interpretation of adversarial attacks from a graph perspective. For an image, benign or adversarial, we study how a neural network's architecture can induce an associated graph. We study this graph and introduce specific measures used to predict and interpret adversarial attacks. We show that graphs-based approaches help to investigate the inner workings of adversarial attacks.



## **14. UKP-SQuARE: An Interactive Tool for Teaching Question Answering**

cs.CL

Accepted by BEA workshop, ACL2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19748v1) [paper-pdf](http://arxiv.org/pdf/2305.19748v1)

**Authors**: Haishuo Fang, Haritz Puerto, Iryna Gurevych

**Abstract**: The exponential growth of question answering (QA) has made it an indispensable topic in any Natural Language Processing (NLP) course. Additionally, the breadth of QA derived from this exponential growth makes it an ideal scenario for teaching related NLP topics such as information retrieval, explainability, and adversarial attacks among others. In this paper, we introduce UKP-SQuARE as a platform for QA education. This platform provides an interactive environment where students can run, compare, and analyze various QA models from different perspectives, such as general behavior, explainability, and robustness. Therefore, students can get a first-hand experience in different QA techniques during the class. Thanks to this, we propose a learner-centered approach for QA education in which students proactively learn theoretical concepts and acquire problem-solving skills through interactive exploration, experimentation, and practical assignments, rather than solely relying on traditional lectures. To evaluate the effectiveness of UKP-SQuARE in teaching scenarios, we adopted it in a postgraduate NLP course and surveyed the students after the course. Their positive feedback shows the platform's effectiveness in their course and invites a wider adoption.



## **15. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2209.01962v5) [paper-pdf](http://arxiv.org/pdf/2209.01962v5)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at https://youtu.be/zJZ1aNlXsMU.



## **16. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

Accepted by IEEE Intelligent Vehicle Symposium, 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2103.09151v7) [paper-pdf](http://arxiv.org/pdf/2103.09151v7)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstract**: As research in deep neural networks advances, deep convolutional networks become promising for autonomous driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for autonomous driving. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. While for regression tasks, the effect of adversarial attacks is not as well understood. In this research, we devise two white-box targeted attacks against end-to-end autonomous driving models. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. In an average of 800 attacks with the same attack strength (epsilon=1), the image-specific and image-agnostic attack deviates the steering angle from the original output by 0.478 and 0.111, respectively, which is much stronger than random noises that only perturbs the steering angle by 0.002 (The steering angle ranges from [-1, 1]). Both attacks can be initiated in real-time on CPUs without employing GPUs. Demo video: https://youtu.be/I0i8uN2oOP0.



## **17. IBP Regularization for Verified Adversarial Robustness via Branch-and-Bound**

cs.LG

ICML 2022 Workshop on Formal Verification of Machine Learning

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2206.14772v2) [paper-pdf](http://arxiv.org/pdf/2206.14772v2)

**Authors**: Alessandro De Palma, Rudy Bunel, Krishnamurthy Dvijotham, M. Pawan Kumar, Robert Stanforth

**Abstract**: Recent works have tried to increase the verifiability of adversarially trained networks by running the attacks over domains larger than the original perturbations and adding various regularization terms to the objective. However, these algorithms either underperform or require complex and expensive stage-wise training procedures, hindering their practical applicability. We present IBP-R, a novel verified training algorithm that is both simple and effective. IBP-R induces network verifiability by coupling adversarial attacks on enlarged domains with a regularization term, based on inexpensive interval bound propagation, that minimizes the gap between the non-convex verification problem and its approximations. By leveraging recent branch-and-bound frameworks, we show that IBP-R obtains state-of-the-art verified robustness-accuracy trade-offs for small perturbations on CIFAR-10 while training significantly faster than relevant previous work. Additionally, we present UPB, a novel branching strategy that, relying on a simple heuristic based on $\beta$-CROWN, reduces the cost of state-of-the-art branching algorithms while yielding splits of comparable quality.



## **18. Adversarial Clean Label Backdoor Attacks and Defenses on Text Classification Systems**

cs.CL

RepL4NLP 2023 at ACL 2023

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19607v1) [paper-pdf](http://arxiv.org/pdf/2305.19607v1)

**Authors**: Ashim Gupta, Amrith Krishna

**Abstract**: Clean-label (CL) attack is a form of data poisoning attack where an adversary modifies only the textual input of the training data, without requiring access to the labeling function. CL attacks are relatively unexplored in NLP, as compared to label flipping (LF) attacks, where the latter additionally requires access to the labeling function as well. While CL attacks are more resilient to data sanitization and manual relabeling methods than LF attacks, they often demand as high as ten times the poisoning budget than LF attacks. In this work, we first introduce an Adversarial Clean Label attack which can adversarially perturb in-class training examples for poisoning the training set. We then show that an adversary can significantly bring down the data requirements for a CL attack, using the aforementioned approach, to as low as 20% of the data otherwise required. We then systematically benchmark and analyze a number of defense methods, for both LF and CL attacks, some previously employed solely for LF attacks in the textual domain and others adapted from computer vision. We find that text-specific defenses greatly vary in their effectiveness depending on their properties.



## **19. Exploring the Vulnerabilities of Machine Learning and Quantum Machine Learning to Adversarial Attacks using a Malware Dataset: A Comparative Analysis**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19593v1) [paper-pdf](http://arxiv.org/pdf/2305.19593v1)

**Authors**: Mst Shapna Akter, Hossain Shahriar, Iysa Iqbal, MD Hossain, M. A. Karim, Victor Clincy, Razvan Voicu

**Abstract**: The burgeoning fields of machine learning (ML) and quantum machine learning (QML) have shown remarkable potential in tackling complex problems across various domains. However, their susceptibility to adversarial attacks raises concerns when deploying these systems in security sensitive applications. In this study, we present a comparative analysis of the vulnerability of ML and QML models, specifically conventional neural networks (NN) and quantum neural networks (QNN), to adversarial attacks using a malware dataset. We utilize a software supply chain attack dataset known as ClaMP and develop two distinct models for QNN and NN, employing Pennylane for quantum implementations and TensorFlow and Keras for traditional implementations. Our methodology involves crafting adversarial samples by introducing random noise to a small portion of the dataset and evaluating the impact on the models performance using accuracy, precision, recall, and F1 score metrics. Based on our observations, both ML and QML models exhibit vulnerability to adversarial attacks. While the QNNs accuracy decreases more significantly compared to the NN after the attack, it demonstrates better performance in terms of precision and recall, indicating higher resilience in detecting true positives under adversarial conditions. We also find that adversarial samples crafted for one model type can impair the performance of the other, highlighting the need for robust defense mechanisms. Our study serves as a foundation for future research focused on enhancing the security and resilience of ML and QML models, particularly QNN, given its recent advancements. A more extensive range of experiments will be conducted to better understand the performance and robustness of both models in the face of adversarial attacks.



## **20. Incremental Randomized Smoothing Certification**

cs.LG

**SubmitDate**: 2023-05-31    [abs](http://arxiv.org/abs/2305.19521v1) [paper-pdf](http://arxiv.org/pdf/2305.19521v1)

**Authors**: Shubham Ugare, Tarun Suresh, Debangshu Banerjee, Gagandeep Singh, Sasa Misailovic

**Abstract**: Randomized smoothing-based certification is an effective approach for obtaining robustness certificates of deep neural networks (DNNs) against adversarial attacks. This method constructs a smoothed DNN model and certifies its robustness through statistical sampling, but it is computationally expensive, especially when certifying with a large number of samples. Furthermore, when the smoothed model is modified (e.g., quantized or pruned), certification guarantees may not hold for the modified DNN, and recertifying from scratch can be prohibitively expensive.   We present the first approach for incremental robustness certification for randomized smoothing, IRS. We show how to reuse the certification guarantees for the original smoothed model to certify an approximated model with very few samples. IRS significantly reduces the computational cost of certifying modified DNNs while maintaining strong robustness guarantees. We experimentally demonstrate the effectiveness of our approach, showing up to 3x certification speedup over the certification that applies randomized smoothing of the approximate model from scratch.



## **21. Pareto Regret Analyses in Multi-objective Multi-armed Bandit**

cs.LG

19 pages; accepted at ICML 2023 and to be published in Proceedings of  Machine Learning Research (PMLR)

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2212.00884v2) [paper-pdf](http://arxiv.org/pdf/2212.00884v2)

**Authors**: Mengfan Xu, Diego Klabjan

**Abstract**: We study Pareto optimality in multi-objective multi-armed bandit by providing a formulation of adversarial multi-objective multi-armed bandit and defining its Pareto regrets that can be applied to both stochastic and adversarial settings. The regrets do not rely on any scalarization functions and reflect Pareto optimality compared to scalarized regrets. We also present new algorithms assuming both with and without prior information of the multi-objective multi-armed bandit setting. The algorithms are shown optimal in adversarial settings and nearly optimal up to a logarithmic factor in stochastic settings simultaneously by our established upper bounds and lower bounds on Pareto regrets. Moreover, the lower bound analyses show that the new regrets are consistent with the existing Pareto regret for stochastic settings and extend an adversarial attack mechanism from bandit to the multi-objective one.



## **22. Adversarial Attacks on Online Learning to Rank with Stochastic Click Models**

cs.LG

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.19218v1) [paper-pdf](http://arxiv.org/pdf/2305.19218v1)

**Authors**: Zichen Wang, Rishab Balasubramanian, Hui Yuan, Chenyu Song, Mengdi Wang, Huazheng Wang

**Abstract**: We propose the first study of adversarial attacks on online learning to rank. The goal of the adversary is to misguide the online learning to rank algorithm to place the target item on top of the ranking list linear times to time horizon $T$ with a sublinear attack cost. We propose generalized list poisoning attacks that perturb the ranking list presented to the user. This strategy can efficiently attack any no-regret ranker in general stochastic click models. Furthermore, we propose a click poisoning-based strategy named attack-then-quit that can efficiently attack two representative OLTR algorithms for stochastic click models. We theoretically analyze the success and cost upper bound of the two proposed methods. Experimental results based on synthetic and real-world data further validate the effectiveness and cost-efficiency of the proposed attack strategies.



## **23. Learning Robust Kernel Ensembles with Kernel Average Pooling**

cs.LG

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2210.00062v2) [paper-pdf](http://arxiv.org/pdf/2210.00062v2)

**Authors**: Pouya Bashivan, Adam Ibrahim, Amirozhan Dehghani, Yifei Ren

**Abstract**: Model ensembles have long been used in machine learning to reduce the variance in individual model predictions, making them more robust to input perturbations. Pseudo-ensemble methods like dropout have also been commonly used in deep learning models to improve generalization. However, the application of these techniques to improve neural networks' robustness against input perturbations remains underexplored. We introduce Kernel Average Pooling (KAP), a neural network building block that applies the mean filter along the kernel dimension of the layer activation tensor. We show that ensembles of kernels with similar functionality naturally emerge in convolutional neural networks equipped with KAP and trained with backpropagation. Moreover, we show that when trained on inputs perturbed with additive Gaussian noise, KAP models are remarkably robust against various forms of adversarial attacks. Empirical evaluations on CIFAR10, CIFAR100, TinyImagenet, and Imagenet datasets show substantial improvements in robustness against strong adversarial attacks such as AutoAttack without training on any adversarial examples.



## **24. Does CLIP Know My Face?**

cs.LG

15 pages, 6 figures

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2209.07341v3) [paper-pdf](http://arxiv.org/pdf/2209.07341v3)

**Authors**: Dominik Hintersdorf, Lukas Struppek, Manuel Brack, Felix Friedrich, Patrick Schramowski, Kristian Kersting

**Abstract**: With the rise of deep learning in various applications, privacy concerns around the protection of training data has become a critical area of research. Whereas prior studies have focused on privacy risks in single-modal models, we introduce a novel method to assess privacy for multi-modal models, specifically vision-language models like CLIP. The proposed Identity Inference Attack (IDIA) reveals whether an individual was included in the training data by querying the model with images of the same person. Letting the model choose from a wide variety of possible text labels, the model reveals whether it recognizes the person and, therefore, was used for training. Our large-scale experiments on CLIP demonstrate that individuals used for training can be identified with very high accuracy. We confirm that the model has learned to associate names with depicted individuals, implying the existence of sensitive information that can be extracted by adversaries. Our results highlight the need for stronger privacy protection in large-scale models and suggest that IDIAs can be used to prove the unauthorized use of data for training and to enforce privacy laws.



## **25. Pseudo-Siamese Network based Timbre-reserved Black-box Adversarial Attack in Speaker Identification**

cs.SD

5 pages

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.19020v1) [paper-pdf](http://arxiv.org/pdf/2305.19020v1)

**Authors**: Qing Wang, Jixun Yao, Ziqian Wang, Pengcheng Guo, Lei Xie

**Abstract**: In this study, we propose a timbre-reserved adversarial attack approach for speaker identification (SID) to not only exploit the weakness of the SID model but also preserve the timbre of the target speaker in a black-box attack setting. Particularly, we generate timbre-reserved fake audio by adding an adversarial constraint during the training of the voice conversion model. Then, we leverage a pseudo-Siamese network architecture to learn from the black-box SID model constraining both intrinsic similarity and structural similarity simultaneously. The intrinsic similarity loss is to learn an intrinsic invariance, while the structural similarity loss is to ensure that the substitute SID model shares a similar decision boundary to the fixed black-box SID model. The substitute model can be used as a proxy to generate timbre-reserved fake audio for attacking. Experimental results on the Audio Deepfake Detection (ADD) challenge dataset indicate that the attack success rate of our proposed approach yields up to 60.58% and 55.38% in the white-box and black-box scenarios, respectively, and can deceive both human beings and machines.



## **26. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

stat.ML

Published in JMLR. Code at https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2202.03397v3) [paper-pdf](http://arxiv.org/pdf/2202.03397v3)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstract**: We analyse a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower-level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise (near) optimal sample complexity. In particular, we propose a simple method which uses (stochastic) fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates.



## **27. Ring Signature from Bonsai Tree: How to Preserve the Long-Term Anonymity**

cs.CR

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2305.16135v2) [paper-pdf](http://arxiv.org/pdf/2305.16135v2)

**Authors**: Mingxing Hu, Yunhong Zhou

**Abstract**: Signer-anonymity is the central feature of ring signatures, which enable a user to sign messages on behalf of an arbitrary set of users, called the ring, without revealing exactly which member of the ring actually generated the signature. Strong and long-term signer-anonymity is a reassuring guarantee for users who are hesitant to leak a secret, especially if the consequences of identification are dire in certain scenarios such as whistleblowing. The notion of \textit{unconditional anonymity}, which protects signer-anonymity even against an infinitely powerful adversary, is considered for ring signatures that aim to achieve long-term signer-anonymity. However, the existing lattice-based works that consider the unconditional anonymity notion did not strictly capture the security requirements imposed in practice, this leads to a realistic attack on signer-anonymity.   In this paper, we present a realistic attack on the unconditional anonymity of ring signatures, and formalize the unconditional anonymity model to strictly capture it. We then propose a lattice-based ring signature construction with unconditional anonymity by leveraging bonsai tree mechanism. Finally, we prove the security in the standard model and demonstrate the unconditional anonymity through both theoretical proof and practical experiments.



## **28. Exploring the Unprecedented Privacy Risks of the Metaverse**

cs.CR

**SubmitDate**: 2023-05-30    [abs](http://arxiv.org/abs/2207.13176v2) [paper-pdf](http://arxiv.org/pdf/2207.13176v2)

**Authors**: Vivek Nair, Gonzalo Munilla Garrido, Dawn Song

**Abstract**: Thirty study participants playtested an innocent-looking "escape room" game in virtual reality (VR). Behind the scenes, an adversarial program had accurately inferred over 25 personal data attributes, from anthropometrics like height and wingspan to demographics like age and gender, within just a few minutes of gameplay. As notoriously data-hungry companies become increasingly involved in VR development, this experimental scenario may soon represent a typical VR user experience. While virtual telepresence applications (and the so-called "metaverse") have recently received increased attention and investment from major tech firms, these environments remain relatively under-studied from a security and privacy standpoint. In this work, we illustrate how VR attackers can covertly ascertain dozens of personal data attributes from seemingly-anonymous users of popular metaverse applications like VRChat. These attackers can be as simple as other VR users without special privilege, and the potential scale and scope of this data collection far exceed what is feasible within traditional mobile and web applications. We aim to shed light on the unique privacy risks of the metaverse, and provide the first holistic framework for understanding intrusive data harvesting attacks in these emerging VR ecosystems.



## **29. UMD: Unsupervised Model Detection for X2X Backdoor Attacks**

cs.LG

ICML 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18651v1) [paper-pdf](http://arxiv.org/pdf/2305.18651v1)

**Authors**: Zhen Xiang, Zidi Xiong, Bo Li

**Abstract**: Backdoor (Trojan) attack is a common threat to deep neural networks, where samples from one or more source classes embedded with a backdoor trigger will be misclassified to adversarial target classes. Existing methods for detecting whether a classifier is backdoor attacked are mostly designed for attacks with a single adversarial target (e.g., all-to-one attack). To the best of our knowledge, without supervision, no existing methods can effectively address the more general X2X attack with an arbitrary number of source classes, each paired with an arbitrary target class. In this paper, we propose UMD, the first Unsupervised Model Detection method that effectively detects X2X backdoor attacks via a joint inference of the adversarial (source, target) class pairs. In particular, we first define a novel transferability statistic to measure and select a subset of putative backdoor class pairs based on a proposed clustering approach. Then, these selected class pairs are jointly assessed based on an aggregation of their reverse-engineered trigger size for detection inference, using a robust and unsupervised anomaly detector we proposed. We conduct comprehensive evaluations on CIFAR-10, GTSRB, and Imagenette dataset, and show that our unsupervised UMD outperforms SOTA detectors (even with supervision) by 17%, 4%, and 8%, respectively, in terms of the detection accuracy against diverse X2X attacks. We also show the strong detection performance of UMD against several strong adaptive attacks.



## **30. Securing Cloud File Systems using Shielded Execution**

cs.CR

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18639v1) [paper-pdf](http://arxiv.org/pdf/2305.18639v1)

**Authors**: Quinn Burke, Yohan Beugin, Blaine Hoak, Rachel King, Eric Pauley, Ryan Sheatsley, Mingli Yu, Ting He, Thomas La Porta, Patrick McDaniel

**Abstract**: Cloud file systems offer organizations a scalable and reliable file storage solution. However, cloud file systems have become prime targets for adversaries, and traditional designs are not equipped to protect organizations against the myriad of attacks that may be initiated by a malicious cloud provider, co-tenant, or end-client. Recently proposed designs leveraging cryptographic techniques and trusted execution environments (TEEs) still force organizations to make undesirable trade-offs, consequently leading to either security, functional, or performance limitations. In this paper, we introduce TFS, a cloud file system that leverages the security capabilities provided by TEEs to bootstrap new security protocols that meet real-world security, functional, and performance requirements. Through extensive security and performance analyses, we show that TFS can ensure stronger security guarantees while still providing practical utility and performance w.r.t. state-of-the-art systems; compared to the widely-used NFS, TFS achieves up to 2.1X speedups across micro-benchmarks and incurs <1X overhead for most macro-benchmark workloads. TFS demonstrates that organizations need not sacrifice file system security to embrace the functional and performance advantages of outsourcing.



## **31. Exploiting Explainability to Design Adversarial Attacks and Evaluate Attack Resilience in Hate-Speech Detection Models**

cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18585v1) [paper-pdf](http://arxiv.org/pdf/2305.18585v1)

**Authors**: Pranath Reddy Kumbam, Sohaib Uddin Syed, Prashanth Thamminedi, Suhas Harish, Ian Perera, Bonnie J. Dorr

**Abstract**: The advent of social media has given rise to numerous ethical challenges, with hate speech among the most significant concerns. Researchers are attempting to tackle this problem by leveraging hate-speech detection and employing language models to automatically moderate content and promote civil discourse. Unfortunately, recent studies have revealed that hate-speech detection systems can be misled by adversarial attacks, raising concerns about their resilience. While previous research has separately addressed the robustness of these models under adversarial attacks and their interpretability, there has been no comprehensive study exploring their intersection. The novelty of our work lies in combining these two critical aspects, leveraging interpretability to identify potential vulnerabilities and enabling the design of targeted adversarial attacks. We present a comprehensive and comparative analysis of adversarial robustness exhibited by various hate-speech detection models. Our study evaluates the resilience of these models against adversarial attacks using explainability techniques. To gain insights into the models' decision-making processes, we employ the Local Interpretable Model-agnostic Explanations (LIME) framework. Based on the explainability results obtained by LIME, we devise and execute targeted attacks on the text by leveraging the TextAttack tool. Our findings enhance the understanding of the vulnerabilities and strengths exhibited by state-of-the-art hate-speech detection models. This work underscores the importance of incorporating explainability in the development and evaluation of such models to enhance their resilience against adversarial attacks. Ultimately, this work paves the way for creating more robust and reliable hate-speech detection systems, fostering safer online environments and promoting ethical discourse on social media platforms.



## **32. Robust Lipschitz Bandits to Adversarial Corruptions**

cs.LG

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18543v1) [paper-pdf](http://arxiv.org/pdf/2305.18543v1)

**Authors**: Yue Kang, Cho-Jui Hsieh, Thomas C. M. Lee

**Abstract**: Lipschitz bandit is a variant of stochastic bandits that deals with a continuous arm set defined on a metric space, where the reward function is subject to a Lipschitz constraint. In this paper, we introduce a new problem of Lipschitz bandits in the presence of adversarial corruptions where an adaptive adversary corrupts the stochastic rewards up to a total budget $C$. The budget is measured by the sum of corruption levels across the time horizon $T$. We consider both weak and strong adversaries, where the weak adversary is unaware of the current action before the attack, while the strong one can observe it. Our work presents the first line of robust Lipschitz bandit algorithms that can achieve sub-linear regret under both types of adversary, even when the total budget of corruption $C$ is unrevealed to the agent. We provide a lower bound under each type of adversary, and show that our algorithm is optimal under the strong case. Finally, we conduct experiments to illustrate the effectiveness of our algorithms against two classic kinds of attacks.



## **33. BITE: Textual Backdoor Attacks with Iterative Trigger Injection**

cs.CL

Accepted to ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2205.12700v3) [paper-pdf](http://arxiv.org/pdf/2205.12700v3)

**Authors**: Jun Yan, Vansh Gupta, Xiang Ren

**Abstract**: Backdoor attacks have become an emerging threat to NLP systems. By providing poisoned training data, the adversary can embed a "backdoor" into the victim model, which allows input instances satisfying certain textual patterns (e.g., containing a keyword) to be predicted as a target label of the adversary's choice. In this paper, we demonstrate that it is possible to design a backdoor attack that is both stealthy (i.e., hard to notice) and effective (i.e., has a high attack success rate). We propose BITE, a backdoor attack that poisons the training data to establish strong correlations between the target label and a set of "trigger words". These trigger words are iteratively identified and injected into the target-label instances through natural word-level perturbations. The poisoned training data instruct the victim model to predict the target label on inputs containing trigger words, forming the backdoor. Experiments on four text classification datasets show that our proposed attack is significantly more effective than baseline methods while maintaining decent stealthiness, raising alarm on the usage of untrusted training data. We further propose a defense method named DeBITE based on potential trigger word removal, which outperforms existing methods in defending against BITE and generalizes well to handling other backdoor attacks.



## **34. Fundamental Limitations of Alignment in Large Language Models**

cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2304.11082v2) [paper-pdf](http://arxiv.org/pdf/2304.11082v2)

**Authors**: Yotam Wolf, Noam Wies, Oshri Avnery, Yoav Levine, Amnon Shashua

**Abstract**: An important aspect in developing language models that interact with humans is aligning their behavior to be useful and unharmful for their human users. This is usually achieved by tuning the model in a way that enhances desired behaviors and inhibits undesired ones, a process referred to as alignment. In this paper, we propose a theoretical approach called Behavior Expectation Bounds (BEB) which allows us to formally investigate several inherent characteristics and limitations of alignment in large language models. Importantly, we prove that for any behavior that has a finite probability of being exhibited by the model, there exist prompts that can trigger the model into outputting this behavior, with probability that increases with the length of the prompt. This implies that any alignment process that attenuates undesired behavior but does not remove it altogether, is not safe against adversarial prompting attacks. Furthermore, our framework hints at the mechanism by which leading alignment approaches such as reinforcement learning from human feedback increase the LLM's proneness to being prompted into the undesired behaviors. Moreover, we include the notion of personas in our BEB framework, and find that behaviors which are generally very unlikely to be exhibited by the model can be brought to the front by prompting the model to behave as specific persona. This theoretical result is being experimentally demonstrated in large scale by the so called contemporary "chatGPT jailbreaks", where adversarial users trick the LLM into breaking its alignment guardrails by triggering it into acting as a malicious persona. Our results expose fundamental limitations in alignment of LLMs and bring to the forefront the need to devise reliable mechanisms for ensuring AI safety.



## **35. From Adversarial Arms Race to Model-centric Evaluation: Motivating a Unified Automatic Robustness Evaluation Framework**

cs.CL

Accepted to Findings of ACL 2023

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18503v1) [paper-pdf](http://arxiv.org/pdf/2305.18503v1)

**Authors**: Yangyi Chen, Hongcheng Gao, Ganqu Cui, Lifan Yuan, Dehan Kong, Hanlu Wu, Ning Shi, Bo Yuan, Longtao Huang, Hui Xue, Zhiyuan Liu, Maosong Sun, Heng Ji

**Abstract**: Textual adversarial attacks can discover models' weaknesses by adding semantic-preserved but misleading perturbations to the inputs. The long-lasting adversarial attack-and-defense arms race in Natural Language Processing (NLP) is algorithm-centric, providing valuable techniques for automatic robustness evaluation. However, the existing practice of robustness evaluation may exhibit issues of incomprehensive evaluation, impractical evaluation protocol, and invalid adversarial samples. In this paper, we aim to set up a unified automatic robustness evaluation framework, shifting towards model-centric evaluation to further exploit the advantages of adversarial attacks. To address the above challenges, we first determine robustness evaluation dimensions based on model capabilities and specify the reasonable algorithm to generate adversarial samples for each dimension. Then we establish the evaluation protocol, including evaluation settings and metrics, under realistic demands. Finally, we use the perturbation degree of adversarial samples to control the sample validity. We implement a toolkit RobTest that realizes our automatic robustness evaluation framework. In our experiments, we conduct a robustness evaluation of RoBERTa models to demonstrate the effectiveness of our evaluation framework, and further show the rationality of each component in the framework. The code will be made public at \url{https://github.com/thunlp/RobTest}.



## **36. Fourier Analysis on Robustness of Graph Convolutional Neural Networks for Skeleton-based Action Recognition**

cs.CV

17 pages, 13 figures

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17939v1) [paper-pdf](http://arxiv.org/pdf/2305.17939v1)

**Authors**: Nariki Tanaka, Hiroshi Kera, Kazuhiko Kawamoto

**Abstract**: Using Fourier analysis, we explore the robustness and vulnerability of graph convolutional neural networks (GCNs) for skeleton-based action recognition. We adopt a joint Fourier transform (JFT), a combination of the graph Fourier transform (GFT) and the discrete Fourier transform (DFT), to examine the robustness of adversarially-trained GCNs against adversarial attacks and common corruptions. Experimental results with the NTU RGB+D dataset reveal that adversarial training does not introduce a robustness trade-off between adversarial attacks and low-frequency perturbations, which typically occurs during image classification based on convolutional neural networks. This finding indicates that adversarial training is a practical approach to enhancing robustness against adversarial attacks and common corruptions in skeleton-based action recognition. Furthermore, we find that the Fourier approach cannot explain vulnerability against skeletal part occlusion corruption, which highlights its limitations. These findings extend our understanding of the robustness of GCNs, potentially guiding the development of more robust learning methods for skeleton-based action recognition.



## **37. Membership Inference Attacks against Language Models via Neighbourhood Comparison**

cs.CL

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.18462v1) [paper-pdf](http://arxiv.org/pdf/2305.18462v1)

**Authors**: Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schölkopf, Mrinmaya Sachan, Taylor Berg-Kirkpatrick

**Abstract**: Membership Inference attacks (MIAs) aim to predict whether a data sample was present in the training data of a machine learning model or not, and are widely used for assessing the privacy risks of language models. Most existing attacks rely on the observation that models tend to assign higher probabilities to their training samples than non-training points. However, simple thresholding of the model score in isolation tends to lead to high false-positive rates as it does not account for the intrinsic complexity of a sample. Recent work has demonstrated that reference-based attacks which compare model scores to those obtained from a reference model trained on similar data can substantially improve the performance of MIAs. However, in order to train reference models, attacks of this kind make the strong and arguably unrealistic assumption that an adversary has access to samples closely resembling the original training data. Therefore, we investigate their performance in more realistic scenarios and find that they are highly fragile in relation to the data distribution used to train reference models. To investigate whether this fragility provides a layer of safety, we propose and evaluate neighbourhood attacks, which compare model scores for a given sample to scores of synthetically generated neighbour texts and therefore eliminate the need for access to the training data distribution. We show that, in addition to being competitive with reference-based attacks that have perfect knowledge about the training data distribution, our attack clearly outperforms existing reference-free attacks as well as reference-based attacks with imperfect knowledge, which demonstrates the need for a reevaluation of the threat model of adversarial attacks.



## **38. NaturalFinger: Generating Natural Fingerprint with Generative Adversarial Networks**

cs.CV

**SubmitDate**: 2023-05-29    [abs](http://arxiv.org/abs/2305.17868v1) [paper-pdf](http://arxiv.org/pdf/2305.17868v1)

**Authors**: Kang Yang, Kunhao Lai

**Abstract**: Deep neural network (DNN) models have become a critical asset of the model owner as training them requires a large amount of resource (i.e. labeled data). Therefore, many fingerprinting schemes have been proposed to safeguard the intellectual property (IP) of the model owner against model extraction and illegal redistribution. However, previous schemes adopt unnatural images as the fingerprint, such as adversarial examples and noisy images, which can be easily perceived and rejected by the adversary. In this paper, we propose NaturalFinger which generates natural fingerprint with generative adversarial networks (GANs). Besides, our proposed NaturalFinger fingerprints the decision difference areas rather than the decision boundary, which is more robust. The application of GAN not only allows us to generate more imperceptible samples, but also enables us to generate unrestricted samples to explore the decision boundary.To demonstrate the effectiveness of our fingerprint approach, we evaluate our approach against four model modification attacks including adversarial training and two model extraction attacks. Experiments show that our approach achieves 0.91 ARUC value on the FingerBench dataset (154 models), exceeding the optimal baseline (MetaV) over 17\%.



## **39. NOTABLE: Transferable Backdoor Attacks Against Prompt-based NLP Models**

cs.CL

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17826v1) [paper-pdf](http://arxiv.org/pdf/2305.17826v1)

**Authors**: Kai Mei, Zheng Li, Zhenting Wang, Yang Zhang, Shiqing Ma

**Abstract**: Prompt-based learning is vulnerable to backdoor attacks. Existing backdoor attacks against prompt-based models consider injecting backdoors into the entire embedding layers or word embedding vectors. Such attacks can be easily affected by retraining on downstream tasks and with different prompting strategies, limiting the transferability of backdoor attacks. In this work, we propose transferable backdoor attacks against prompt-based models, called NOTABLE, which is independent of downstream tasks and prompting strategies. Specifically, NOTABLE injects backdoors into the encoders of PLMs by utilizing an adaptive verbalizer to bind triggers to specific words (i.e., anchors). It activates the backdoor by pasting input with triggers to reach adversary-desired anchors, achieving independence from downstream tasks and prompting strategies. We conduct experiments on six NLP tasks, three popular models, and three prompting strategies. Empirical results show that NOTABLE achieves superior attack performance (i.e., attack success rate over 90% on all the datasets), and outperforms two state-of-the-art baselines. Evaluations on three defenses show the robustness of NOTABLE. Our code can be found at https://github.com/RU-System-Software-and-Security/Notable.



## **40. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

cs.CV

Code will be available at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.13625v2) [paper-pdf](http://arxiv.org/pdf/2305.13625v2)

**Authors**: Jiang Liu, Chun Pong Lau, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.



## **41. Amplification trojan network: Attack deep neural networks by amplifying their inherent weakness**

cs.CR

Published Sep 2022 in Neurocomputing

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.17688v1) [paper-pdf](http://arxiv.org/pdf/2305.17688v1)

**Authors**: Zhanhao Hu, Jun Zhu, Bo Zhang, Xiaolin Hu

**Abstract**: Recent works found that deep neural networks (DNNs) can be fooled by adversarial examples, which are crafted by adding adversarial noise on clean inputs. The accuracy of DNNs on adversarial examples will decrease as the magnitude of the adversarial noise increase. In this study, we show that DNNs can be also fooled when the noise is very small under certain circumstances. This new type of attack is called Amplification Trojan Attack (ATAttack). Specifically, we use a trojan network to transform the inputs before sending them to the target DNN. This trojan network serves as an amplifier to amplify the inherent weakness of the target DNN. The target DNN, which is infected by the trojan network, performs normally on clean data while being more vulnerable to adversarial examples. Since it only transforms the inputs, the trojan network can hide in DNN-based pipelines, e.g. by infecting the pre-processing procedure of the inputs before sending them to the DNNs. This new type of threat should be considered in developing safe DNNs.



## **42. Threat Models over Space and Time: A Case Study of E2EE Messaging Applications**

cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2301.05653v2) [paper-pdf](http://arxiv.org/pdf/2301.05653v2)

**Authors**: Partha Das Chowdhury, Maria Sameen, Jenny Blessing, Nicholas Boucher, Joseph Gardiner, Tom Burrows, Ross Anderson, Awais Rashid

**Abstract**: Threat modelling is foundational to secure systems engineering and should be done in consideration of the context within which systems operate. On the other hand, the continuous evolution of both the technical sophistication of threats and the system attack surface is an inescapable reality. In this work, we explore the extent to which real-world systems engineering reflects the changing threat context. To this end we examine the desktop clients of six widely used end-to-end-encrypted mobile messaging applications to understand the extent to which they adjusted their threat model over space (when enabling clients on new platforms, such as desktop clients) and time (as new threats emerged). We experimented with short-lived adversarial access against these desktop clients and analyzed the results with respect to two popular threat elicitation frameworks, STRIDE and LINDDUN. The results demonstrate that system designers need to both recognise the threats in the evolving context within which systems operate and, more importantly, to mitigate them by rescoping trust boundaries in a manner that those within the administrative boundary cannot violate security and privacy properties. Such a nuanced understanding of trust boundary scopes and their relationship with administrative boundaries allows for better administration of shared components, including securing them with safe defaults.



## **43. A Synergistic Framework Leveraging Autoencoders and Generative Adversarial Networks for the Synthesis of Computational Fluid Dynamics Results in Aerofoil Aerodynamics**

physics.flu-dyn

9 pages, 11 figures

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18386v1) [paper-pdf](http://arxiv.org/pdf/2305.18386v1)

**Authors**: Tanishk Nandal, Vaibhav Fulara, Raj Kumar Singh

**Abstract**: In the realm of computational fluid dynamics (CFD), accurate prediction of aerodynamic behaviour plays a pivotal role in aerofoil design and optimization. This study proposes a novel approach that synergistically combines autoencoders and Generative Adversarial Networks (GANs) for the purpose of generating CFD results. Our innovative framework harnesses the intrinsic capabilities of autoencoders to encode aerofoil geometries into a compressed and informative 20-length vector representation. Subsequently, a conditional GAN network adeptly translates this vector into precise pressure-distribution plots, accounting for fixed wind velocity, angle of attack, and turbulence level specifications. The training process utilizes a meticulously curated dataset acquired from JavaFoil software, encompassing a comprehensive range of aerofoil geometries. The proposed approach exhibits profound potential in reducing the time and costs associated with aerodynamic prediction, enabling efficient evaluation of aerofoil performance. The findings contribute to the advancement of computational techniques in fluid dynamics and pave the way for enhanced design and optimization processes in aerodynamics.



## **44. Backdoor Attacks Against Incremental Learners: An Empirical Evaluation Study**

cs.CR

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18384v1) [paper-pdf](http://arxiv.org/pdf/2305.18384v1)

**Authors**: Yiqi Zhong, Xianming Liu, Deming Zhai, Junjun Jiang, Xiangyang Ji

**Abstract**: Large amounts of incremental learning algorithms have been proposed to alleviate the catastrophic forgetting issue arises while dealing with sequential data on a time series. However, the adversarial robustness of incremental learners has not been widely verified, leaving potential security risks. Specifically, for poisoning-based backdoor attacks, we argue that the nature of streaming data in IL provides great convenience to the adversary by creating the possibility of distributed and cross-task attacks -- an adversary can affect \textbf{any unknown} previous or subsequent task by data poisoning \textbf{at any time or time series} with extremely small amount of backdoor samples injected (e.g., $0.1\%$ based on our observations). To attract the attention of the research community, in this paper, we empirically reveal the high vulnerability of 11 typical incremental learners against poisoning-based backdoor attack on 3 learning scenarios, especially the cross-task generalization effect of backdoor knowledge, while the poison ratios range from $5\%$ to as low as $0.1\%$. Finally, the defense mechanism based on activation clustering is found to be effective in detecting our trigger pattern to mitigate potential security risks.



## **45. BadLabel: A Robust Perspective on Evaluating and Enhancing Label-noise Learning**

cs.LG

**SubmitDate**: 2023-05-28    [abs](http://arxiv.org/abs/2305.18377v1) [paper-pdf](http://arxiv.org/pdf/2305.18377v1)

**Authors**: Jingfeng Zhang, Bo Song, Haohan Wang, Bo Han, Tongliang Liu, Lei Liu, Masashi Sugiyama

**Abstract**: Label-noise learning (LNL) aims to increase the model's generalization given training data with noisy labels. To facilitate practical LNL algorithms, researchers have proposed different label noise types, ranging from class-conditional to instance-dependent noises. In this paper, we introduce a novel label noise type called BadLabel, which can significantly degrade the performance of existing LNL algorithms by a large margin. BadLabel is crafted based on the label-flipping attack against standard classification, where specific samples are selected and their labels are flipped to other labels so that the loss values of clean and noisy labels become indistinguishable. To address the challenge posed by BadLabel, we further propose a robust LNL method that perturbs the labels in an adversarial manner at each epoch to make the loss values of clean and noisy labels again distinguishable. Once we select a small set of (mostly) clean labeled data, we can apply the techniques of semi-supervised learning to train the model accurately. Empirically, our experimental results demonstrate that existing LNL algorithms are vulnerable to the newly introduced BadLabel noise type, while our proposed robust LNL method can effectively improve the generalization performance of the model under various types of label noise. The new dataset of noisy labels and the source codes of robust LNL algorithms are available at https://github.com/zjfheart/BadLabels.



## **46. SneakyPrompt: Evaluating Robustness of Text-to-image Generative Models' Safety Filters**

cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.12082v2) [paper-pdf](http://arxiv.org/pdf/2305.12082v2)

**Authors**: Yuchen Yang, Bo Hui, Haolin Yuan, Neil Gong, Yinzhi Cao

**Abstract**: Text-to-image generative models such as Stable Diffusion and DALL$\cdot$E 2 have attracted much attention since their publication due to their wide application in the real world. One challenging problem of text-to-image generative models is the generation of Not-Safe-for-Work (NSFW) content, e.g., those related to violence and adult. Therefore, a common practice is to deploy a so-called safety filter, which blocks NSFW content based on either text or image features. Prior works have studied the possible bypass of such safety filters. However, existing works are largely manual and specific to Stable Diffusion's official safety filter. Moreover, the bypass ratio of Stable Diffusion's safety filter is as low as 23.51% based on our evaluation.   In this paper, we propose the first automated attack framework, called SneakyPrompt, to evaluate the robustness of real-world safety filters in state-of-the-art text-to-image generative models. Our key insight is to search for alternative tokens in a prompt that generates NSFW images so that the generated prompt (called an adversarial prompt) bypasses existing safety filters. Specifically, SneakyPrompt utilizes reinforcement learning (RL) to guide an agent with positive rewards on semantic similarity and bypass success.   Our evaluation shows that SneakyPrompt successfully generated NSFW content using an online model DALL$\cdot$E 2 with its default, closed-box safety filter enabled. At the same time, we also deploy several open-source state-of-the-art safety filters on a Stable Diffusion model and show that SneakyPrompt not only successfully generates NSFW content, but also outperforms existing adversarial attacks in terms of the number of queries and image qualities.



## **47. Tubes Among Us: Analog Attack on Automatic Speaker Identification**

cs.LG

Published at USENIX Security 2023  https://www.usenix.org/conference/usenixsecurity23/presentation/ahmed

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2202.02751v2) [paper-pdf](http://arxiv.org/pdf/2202.02751v2)

**Authors**: Shimaa Ahmed, Yash Wani, Ali Shahin Shamsabadi, Mohammad Yaghini, Ilia Shumailov, Nicolas Papernot, Kassem Fawaz

**Abstract**: Recent years have seen a surge in the popularity of acoustics-enabled personal devices powered by machine learning. Yet, machine learning has proven to be vulnerable to adversarial examples. A large number of modern systems protect themselves against such attacks by targeting artificiality, i.e., they deploy mechanisms to detect the lack of human involvement in generating the adversarial examples. However, these defenses implicitly assume that humans are incapable of producing meaningful and targeted adversarial examples. In this paper, we show that this base assumption is wrong. In particular, we demonstrate that for tasks like speaker identification, a human is capable of producing analog adversarial examples directly with little cost and supervision: by simply speaking through a tube, an adversary reliably impersonates other speakers in eyes of ML models for speaker identification. Our findings extend to a range of other acoustic-biometric tasks such as liveness detection, bringing into question their use in security-critical settings in real life, such as phone banking.



## **48. PowerGAN: A Machine Learning Approach for Power Side-Channel Attack on Compute-in-Memory Accelerators**

cs.CR

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2304.11056v2) [paper-pdf](http://arxiv.org/pdf/2304.11056v2)

**Authors**: Ziyu Wang, Yuting Wu, Yongmo Park, Sangmin Yoo, Xinxin Wang, Jason K. Eshraghian, Wei D. Lu

**Abstract**: Analog compute-in-memory (CIM) systems are promising for deep neural network (DNN) inference acceleration due to their energy efficiency and high throughput. However, as the use of DNNs expands, protecting user input privacy has become increasingly important. In this paper, we identify a potential security vulnerability wherein an adversary can reconstruct the user's private input data from a power side-channel attack, under proper data acquisition and pre-processing, even without knowledge of the DNN model. We further demonstrate a machine learning-based attack approach using a generative adversarial network (GAN) to enhance the data reconstruction. Our results show that the attack methodology is effective in reconstructing user inputs from analog CIM accelerator power leakage, even at large noise levels and after countermeasures are applied. Specifically, we demonstrate the efficacy of our approach on an example of U-Net inference chip for brain tumor detection, and show the original magnetic resonance imaging (MRI) medical images can be successfully reconstructed even at a noise-level of 20% standard deviation of the maximum power signal value. Our study highlights a potential security vulnerability in analog CIM accelerators and raises awareness of using GAN to breach user privacy in such systems.



## **49. Two Heads are Better than One: Towards Better Adversarial Robustness by Combining Transduction and Rejection**

cs.LG

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17528v1) [paper-pdf](http://arxiv.org/pdf/2305.17528v1)

**Authors**: Nils Palumbo, Yang Guo, Xi Wu, Jiefeng Chen, Yingyu Liang, Somesh Jha

**Abstract**: Both transduction and rejection have emerged as important techniques for defending against adversarial perturbations. A recent work by Tram\`er showed that, in the rejection-only case (no transduction), a strong rejection-solution can be turned into a strong (but computationally inefficient) non-rejection solution. This detector-to-classifier reduction has been mostly applied to give evidence that certain claims of strong selective-model solutions are susceptible, leaving the benefits of rejection unclear. On the other hand, a recent work by Goldwasser et al. showed that rejection combined with transduction can give provable guarantees (for certain problems) that cannot be achieved otherwise. Nevertheless, under recent strong adversarial attacks (GMSA, which has been shown to be much more effective than AutoAttack against transduction), Goldwasser et al.'s work was shown to have low performance in a practical deep-learning setting. In this paper, we take a step towards realizing the promise of transduction+rejection in more realistic scenarios. Theoretically, we show that a novel application of Tram\`er's classifier-to-detector technique in the transductive setting can give significantly improved sample-complexity for robust generalization. While our theoretical construction is computationally inefficient, it guides us to identify an efficient transductive algorithm to learn a selective model. Extensive experiments using state of the art attacks (AutoAttack, GMSA) show that our solutions provide significantly better robust accuracy.



## **50. Backdooring Neural Code Search**

cs.SE

**SubmitDate**: 2023-05-27    [abs](http://arxiv.org/abs/2305.17506v1) [paper-pdf](http://arxiv.org/pdf/2305.17506v1)

**Authors**: Weisong Sun, Yuchen Chen, Guanhong Tao, Chunrong Fang, Xiangyu Zhang, Quanjun Zhang, Bin Luo

**Abstract**: Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our attack BADCODE features a special trigger generation and injection procedure, making the attack more effective and stealthy. The evaluation is conducted on two neural code search models and the results show our attack outperforms baselines by 60%. Our user study demonstrates that our attack is more stealthy than the baseline by two times based on the F1 score.



