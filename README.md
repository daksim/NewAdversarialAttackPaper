# Latest Adversarial Attack Papers
**update at 2022-12-02 19:32:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Purifier: Defending Data Inference Attacks via Transforming Confidence Scores**

cs.LG

accepted by AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00612v1) [paper-pdf](http://arxiv.org/pdf/2212.00612v1)

**Authors**: Ziqi Yang, Lijin Wang, Da Yang, Jie Wan, Ziming Zhao, Ee-Chien Chang, Fan Zhang, Kui Ren

**Abstract**: Neural networks are susceptible to data inference attacks such as the membership inference attack, the adversarial model inversion attack and the attribute inference attack, where the attacker could infer useful information such as the membership, the reconstruction or the sensitive attributes of a data sample from the confidence scores predicted by the target classifier. In this paper, we propose a method, namely PURIFIER, to defend against membership inference attacks. It transforms the confidence score vectors predicted by the target classifier and makes purified confidence scores indistinguishable in individual shape, statistical distribution and prediction label between members and non-members. The experimental results show that PURIFIER helps defend membership inference attacks with high effectiveness and efficiency, outperforming previous defense methods, and also incurs negligible utility loss. Besides, our further experiments show that PURIFIER is also effective in defending adversarial model inversion attacks and attribute inference attacks. For example, the inversion error is raised about 4+ times on the Facescrub530 classifier, and the attribute inference accuracy drops significantly when PURIFIER is deployed in our experiment.



## **2. PointCA: Evaluating the Robustness of 3D Point Cloud Completion Models Against Adversarial Examples**

cs.CV

Accepted by the 37th AAAI Conference on Artificial Intelligence  (AAAI-23)

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2211.12294v2) [paper-pdf](http://arxiv.org/pdf/2211.12294v2)

**Authors**: Shengshan Hu, Junwei Zhang, Wei Liu, Junhui Hou, Minghui Li, Leo Yu Zhang, Hai Jin, Lichao Sun

**Abstract**: Point cloud completion, as the upstream procedure of 3D recognition and segmentation, has become an essential part of many tasks such as navigation and scene understanding. While various point cloud completion models have demonstrated their powerful capabilities, their robustness against adversarial attacks, which have been proven to be fatally malicious towards deep neural networks, remains unknown. In addition, existing attack approaches towards point cloud classifiers cannot be applied to the completion models due to different output forms and attack purposes. In order to evaluate the robustness of the completion models, we propose PointCA, the first adversarial attack against 3D point cloud completion models. PointCA can generate adversarial point clouds that maintain high similarity with the original ones, while being completed as another object with totally different semantic information. Specifically, we minimize the representation discrepancy between the adversarial example and the target point set to jointly explore the adversarial point clouds in the geometry space and the feature space. Furthermore, to launch a stealthier attack, we innovatively employ the neighbourhood density information to tailor the perturbation constraint, leading to geometry-aware and distribution-adaptive modifications for each point. Extensive experiments against different premier point cloud completion networks show that PointCA can cause a performance degradation from 77.9% to 16.7%, with the structure chamfer distance kept below 0.01. We conclude that existing completion models are severely vulnerable to adversarial examples, and state-of-the-art defenses for point cloud classification will be partially invalid when applied to incomplete and uneven point cloud data.



## **3. Defending Black-box Skeleton-based Human Activity Classifiers**

cs.CV

Accepted in AAAI 2023

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2203.04713v5) [paper-pdf](http://arxiv.org/pdf/2203.04713v5)

**Authors**: He Wang, Yunfeng Diao, Zichang Tan, Guodong Guo

**Abstract**: Skeletal motions have been heavily replied upon for human activity recognition (HAR). Recently, a universal vulnerability of skeleton-based HAR has been identified across a variety of classifiers and data, calling for mitigation. To this end, we propose the first black-box defense method for skeleton-based HAR to our best knowledge. Our method is featured by full Bayesian treatments of the clean data, the adversaries and the classifier, leading to (1) a new Bayesian Energy-based formulation of robust discriminative classifiers, (2) a new adversary sampling scheme based on natural motion manifolds, and (3) a new post-train Bayesian strategy for black-box defense. We name our framework Bayesian Energy-based Adversarial Training or BEAT. BEAT is straightforward but elegant, which turns vulnerable black-box classifiers into robust ones without sacrificing accuracy. It demonstrates surprising and universal effectiveness across a wide range of skeletal HAR classifiers and datasets, under various attacks. Code is available at https://github.com/realcrane/RobustActionRecogniser.



## **4. All You Need Is Hashing: Defending Against Data Reconstruction Attack in Vertical Federated Learning**

cs.CR

**SubmitDate**: 2022-12-01    [abs](http://arxiv.org/abs/2212.00325v1) [paper-pdf](http://arxiv.org/pdf/2212.00325v1)

**Authors**: Pengyu Qiu, Xuhong Zhang, Shouling Ji, Yuwen Pu, Ting Wang

**Abstract**: Vertical federated learning is a trending solution for multi-party collaboration in training machine learning models. Industrial frameworks adopt secure multi-party computation methods such as homomorphic encryption to guarantee data security and privacy. However, a line of work has revealed that there are still leakage risks in VFL. The leakage is caused by the correlation between the intermediate representations and the raw data. Due to the powerful approximation ability of deep neural networks, an adversary can capture the correlation precisely and reconstruct the data. To deal with the threat of the data reconstruction attack, we propose a hashing-based VFL framework, called \textit{HashVFL}, to cut off the reversibility directly. The one-way nature of hashing allows our framework to block all attempts to recover data from hash codes. However, integrating hashing also brings some challenges, e.g., the loss of information. This paper proposes and addresses three challenges to integrating hashing: learnability, bit balance, and consistency. Experimental results demonstrate \textit{HashVFL}'s efficiency in keeping the main task's performance and defending against data reconstruction attacks. Furthermore, we also analyze its potential value in detecting abnormal inputs. In addition, we conduct extensive experiments to prove \textit{HashVFL}'s generalization in various settings. In summary, \textit{HashVFL} provides a new perspective on protecting multi-party's data security and privacy in VFL. We hope our study can attract more researchers to expand the application domains of \textit{HashVFL}.



## **5. Overcoming the Convex Relaxation Barrier for Neural Network Verification via Nonconvex Low-Rank Semidefinite Relaxations**

cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17244v1) [paper-pdf](http://arxiv.org/pdf/2211.17244v1)

**Authors**: Hong-Ming Chiu, Richard Y. Zhang

**Abstract**: To rigorously certify the robustness of neural networks to adversarial perturbations, most state-of-the-art techniques rely on a triangle-shaped linear programming (LP) relaxation of the ReLU activation. While the LP relaxation is exact for a single neuron, recent results suggest that it faces an inherent "convex relaxation barrier" as additional activations are added, and as the attack budget is increased. In this paper, we propose a nonconvex relaxation for the ReLU relaxation, based on a low-rank restriction of a semidefinite programming (SDP) relaxation. We show that the nonconvex relaxation has a similar complexity to the LP relaxation, but enjoys improved tightness that is comparable to the much more expensive SDP relaxation. Despite nonconvexity, we prove that the verification problem satisfies constraint qualification, and therefore a Riemannian staircase approach is guaranteed to compute a near-globally optimal solution in polynomial time. Our experiments provide evidence that our nonconvex relaxation almost completely overcome the "convex relaxation barrier" faced by the LP relaxation.



## **6. Differentially Private ADMM-Based Distributed Discrete Optimal Transport for Resource Allocation**

cs.SI

6 pages, 4 images, 1 algorithm, IEEE GLOBECOMM 2022

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.17070v1) [paper-pdf](http://arxiv.org/pdf/2211.17070v1)

**Authors**: Jason Hughes, Juntao Chen

**Abstract**: Optimal transport (OT) is a framework that can guide the design of efficient resource allocation strategies in a network of multiple sources and targets. To ease the computational complexity of large-scale transport design, we first develop a distributed algorithm based on the alternating direction method of multipliers (ADMM). However, such a distributed algorithm is vulnerable to sensitive information leakage when an attacker intercepts the transport decisions communicated between nodes during the distributed ADMM updates. To this end, we propose a privacy-preserving distributed mechanism based on output variable perturbation by adding appropriate randomness to each node's decision before it is shared with other corresponding nodes at each update instance. We show that the developed scheme is differentially private, which prevents the adversary from inferring the node's confidential information even knowing the transport decisions. Finally, we corroborate the effectiveness of the devised algorithm through case studies.



## **7. A Systematic Evaluation of Node Embedding Robustness**

cs.LG

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2209.08064v3) [paper-pdf](http://arxiv.org/pdf/2209.08064v3)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstract**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has grown significantly in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring attacks computed using network properties as well as node labels. We also investigate the performance of popular node classification attack baselines that assume full knowledge of the node labels. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We find that node classification results are impacted more than network reconstruction ones, that degree-based and label-based attacks are on average the most damaging and that label heterophily can strongly influence attack performance.



## **8. Adaptive adversarial training method for improving multi-scale GAN based on generalization bound theory**

cs.CV

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2211.16791v1) [paper-pdf](http://arxiv.org/pdf/2211.16791v1)

**Authors**: Jing Tang, Bo Tao, Zeyu Gong, Zhouping Yin

**Abstract**: In recent years, multi-scale generative adversarial networks (GANs) have been proposed to build generalized image processing models based on single sample. Constraining on the sample size, multi-scale GANs have much difficulty converging to the global optimum, which ultimately leads to limitations in their capabilities. In this paper, we pioneered the introduction of PAC-Bayes generalized bound theory into the training analysis of specific models under different adversarial training methods, which can obtain a non-vacuous upper bound on the generalization error for the specified multi-scale GAN structure. Based on the drastic changes we found of the generalization error bound under different adversarial attacks and different training states, we proposed an adaptive training method which can greatly improve the image manipulation ability of multi-scale GANs. The final experimental results show that our adaptive training method in this paper has greatly contributed to the improvement of the quality of the images generated by multi-scale GANs on several image manipulation tasks. In particular, for the image super-resolution restoration task, the multi-scale GAN model trained by the proposed method achieves a 100% reduction in natural image quality evaluator (NIQE) and a 60% reduction in root mean squared error (RMSE), which is better than many models trained on large-scale datasets.



## **9. GARNET: Reduced-Rank Topology Learning for Robust and Scalable Graph Neural Networks**

cs.LG

Published as a conference paper at LoG 2020

**SubmitDate**: 2022-11-30    [abs](http://arxiv.org/abs/2201.12741v3) [paper-pdf](http://arxiv.org/pdf/2201.12741v3)

**Authors**: Chenhui Deng, Xiuyu Li, Zhuo Feng, Zhiru Zhang

**Abstract**: Graph neural networks (GNNs) have been increasingly deployed in various applications that involve learning on non-Euclidean data. However, recent studies show that GNNs are vulnerable to graph adversarial attacks. Although there are several defense methods to improve GNN robustness by eliminating adversarial components, they may also impair the underlying clean graph structure that contributes to GNN training. In addition, few of those defense models can scale to large graphs due to their high computational complexity and memory usage. In this paper, we propose GARNET, a scalable spectral method to boost the adversarial robustness of GNN models. GARNET first leverages weighted spectral embedding to construct a base graph, which is not only resistant to adversarial attacks but also contains critical (clean) graph structure for GNN training. Next, GARNET further refines the base graph by pruning additional uncritical edges based on probabilistic graphical model. GARNET has been evaluated on various datasets, including a large graph with millions of nodes. Our extensive experiment results show that GARNET achieves adversarial accuracy improvement and runtime speedup over state-of-the-art GNN (defense) models by up to 13.27% and 14.7x, respectively.



## **10. Sludge for Good: Slowing and Imposing Costs on Cyber Attackers**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16626v1) [paper-pdf](http://arxiv.org/pdf/2211.16626v1)

**Authors**: Josiah Dykstra, Kelly Shortridge, Jamie Met, Douglas Hough

**Abstract**: Choice architecture describes the design by which choices are presented to people. Nudges are an aspect intended to make "good" outcomes easy, such as using password meters to encourage strong passwords. Sludge, on the contrary, is friction that raises the transaction cost and is often seen as a negative to users. Turning this concept around, we propose applying sludge for positive cybersecurity outcomes by using it offensively to consume attackers' time and other resources.   To date, most cyber defenses have been designed to be optimally strong and effective and prohibit or eliminate attackers as quickly as possible. Our complimentary approach is to also deploy defenses that seek to maximize the consumption of the attackers' time and other resources while causing as little damage as possible to the victim. This is consistent with zero trust and similar mindsets which assume breach. The Sludge Strategy introduces cost-imposing cyber defense by strategically deploying friction for attackers before, during, and after an attack using deception and authentic design features. We present the characteristics of effective sludge, and show a continuum from light to heavy sludge. We describe the quantitative and qualitative costs to attackers and offer practical considerations for deploying sludge in practice. Finally, we examine real-world examples of U.S. government operations to frustrate and impose cost on cyber adversaries.



## **11. Synthesizing Attack-Aware Control and Active Sensing Strategies under Reactive Sensor Attacks**

math.OC

7 pages, 3 figure, 1 table, 1 algorithm

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2204.01584v2) [paper-pdf](http://arxiv.org/pdf/2204.01584v2)

**Authors**: Sumukha Udupa, Abhishek N. Kulkarni, Shuo Han, Nandi O. Leslie, Charles A. Kamhoua, Jie Fu

**Abstract**: We consider the probabilistic planning problem for a defender (P1) who can jointly query the sensors and take control actions to reach a set of goal states while being aware of possible sensor attacks by an adversary (P2) who has perfect observations. To synthesize a provably-correct, attack-aware joint control and active sensing strategy for P1, we construct a stochastic game on graph with augmented states that include the actual game state (known only to the attacker), the belief of the defender about the game state (constructed by the attacker based on his knowledge of defender's observations). We present an algorithm to compute a belief-based, randomized strategy for P1 to ensure satisfying the reachability objective with probability one, under the worst-case sensor attack carried out by an informed P2. We prove the correctness of the algorithm and illustrate using an example.



## **12. Improving Adversarial Robustness with Self-Paced Hard-Class Pair Reweighting**

cs.CV

AAAI-23

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2210.15068v2) [paper-pdf](http://arxiv.org/pdf/2210.15068v2)

**Authors**: Pengyue Hou, Jie Han, Xingyu Li

**Abstract**: Deep Neural Networks are vulnerable to adversarial attacks. Among many defense strategies, adversarial training with untargeted attacks is one of the most effective methods. Theoretically, adversarial perturbation in untargeted attacks can be added along arbitrary directions and the predicted labels of untargeted attacks should be unpredictable. However, we find that the naturally imbalanced inter-class semantic similarity makes those hard-class pairs become virtual targets of each other. This study investigates the impact of such closely-coupled classes on adversarial attacks and develops a self-paced reweighting strategy in adversarial training accordingly. Specifically, we propose to upweight hard-class pair losses in model optimization, which prompts learning discriminative features from hard classes. We further incorporate a term to quantify hard-class pair consistency in adversarial training, which greatly boosts model robustness. Extensive experiments show that the proposed adversarial training method achieves superior robustness performance over state-of-the-art defenses against a wide range of adversarial attacks.



## **13. Resilient Risk based Adaptive Authentication and Authorization (RAD-AA) Framework**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2208.02592v3) [paper-pdf](http://arxiv.org/pdf/2208.02592v3)

**Authors**: Jaimandeep Singh, Chintan Patel, Naveen Kumar Chaudhary

**Abstract**: In recent cyber attacks, credential theft has emerged as one of the primary vectors of gaining entry into the system. Once attacker(s) have a foothold in the system, they use various techniques including token manipulation to elevate the privileges and access protected resources. This makes authentication and token based authorization a critical component for a secure and resilient cyber system. In this paper we discuss the design considerations for such a secure and resilient authentication and authorization framework capable of self-adapting based on the risk scores and trust profiles. We compare this design with the existing standards such as OAuth 2.0, OpenID Connect and SAML 2.0. We then study popular threat models such as STRIDE and PASTA and summarize the resilience of the proposed architecture against common and relevant threat vectors. We call this framework as Resilient Risk based Adaptive Authentication and Authorization (RAD-AA). The proposed framework excessively increases the cost for an adversary to launch and sustain any cyber attack and provides much-needed strength to critical infrastructure. We also discuss the machine learning (ML) approach for the adaptive engine to accurately classify transactions and arrive at risk scores.



## **14. Ada3Diff: Defending against 3D Adversarial Point Clouds via Adaptive Diffusion**

cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16247v1) [paper-pdf](http://arxiv.org/pdf/2211.16247v1)

**Authors**: Kui Zhang, Hang Zhou, Jie Zhang, Qidong Huang, Weiming Zhang, Nenghai Yu

**Abstract**: Deep 3D point cloud models are sensitive to adversarial attacks, which poses threats to safety-critical applications such as autonomous driving. Robust training and defend-by-denoise are typical strategies for defending adversarial perturbations, including adversarial training and statistical filtering, respectively. However, they either induce massive computational overhead or rely heavily upon specified noise priors, limiting generalized robustness against attacks of all kinds. This paper introduces a new defense mechanism based on denoising diffusion models that can adaptively remove diverse noises with a tailored intensity estimator. Specifically, we first estimate adversarial distortions by calculating the distance of the points to their neighborhood best-fit plane. Depending on the distortion degree, we choose specific diffusion time steps for the input point cloud and perform the forward diffusion to disrupt potential adversarial shifts. Then we conduct the reverse denoising process to restore the disrupted point cloud back to a clean distribution. This approach enables effective defense against adaptive attacks with varying noise budgets, achieving accentuated robustness of existing 3D deep recognition models.



## **15. Defending Adversarial Attacks on Deep Learning Based Power Allocation in Massive MIMO Using Denoising Autoencoders**

eess.SP

This work is currently under review for publication

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15365v2) [paper-pdf](http://arxiv.org/pdf/2211.15365v2)

**Authors**: Rajeev Sahay, Minjun Zhang, David J. Love, Christopher G. Brinton

**Abstract**: Recent work has advocated for the use of deep learning to perform power allocation in the downlink of massive MIMO (maMIMO) networks. Yet, such deep learning models are vulnerable to adversarial attacks. In the context of maMIMO power allocation, adversarial attacks refer to the injection of subtle perturbations into the deep learning model's input, during inference (i.e., the adversarial perturbation is injected into inputs during deployment after the model has been trained) that are specifically crafted to force the trained regression model to output an infeasible power allocation solution. In this work, we develop an autoencoder-based mitigation technique, which allows deep learning-based power allocation models to operate in the presence of adversaries without requiring retraining. Specifically, we develop a denoising autoencoder (DAE), which learns a mapping between potentially perturbed data and its corresponding unperturbed input. We test our defense across multiple attacks and in multiple threat models and demonstrate its ability to (i) mitigate the effects of adversarial attacks on power allocation networks using two common precoding schemes, (ii) outperform previously proposed benchmarks for mitigating regression-based adversarial attacks on maMIMO networks, (iii) retain accurate performance in the absence of an attack, and (iv) operate with low computational overhead.



## **16. Quantization-aware Interval Bound Propagation for Training Certifiably Robust Quantized Neural Networks**

cs.LG

Accepted at AAAI 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16187v1) [paper-pdf](http://arxiv.org/pdf/2211.16187v1)

**Authors**: Mathias Lechner, Đorđe Žikelić, Krishnendu Chatterjee, Thomas A. Henzinger, Daniela Rus

**Abstract**: We study the problem of training and certifying adversarially robust quantized neural networks (QNNs). Quantization is a technique for making neural networks more efficient by running them using low-bit integer arithmetic and is therefore commonly adopted in industry. Recent work has shown that floating-point neural networks that have been verified to be robust can become vulnerable to adversarial attacks after quantization, and certification of the quantized representation is necessary to guarantee robustness. In this work, we present quantization-aware interval bound propagation (QA-IBP), a novel method for training robust QNNs. Inspired by advances in robust learning of non-quantized networks, our training algorithm computes the gradient of an abstract representation of the actual network. Unlike existing approaches, our method can handle the discrete semantics of QNNs. Based on QA-IBP, we also develop a complete verification procedure for verifying the adversarial robustness of QNNs, which is guaranteed to terminate and produce a correct answer. Compared to existing approaches, the key advantage of our verification procedure is that it runs entirely on GPU or other accelerator devices. We demonstrate experimentally that our approach significantly outperforms existing methods and establish the new state-of-the-art for training and certifying the robustness of QNNs.



## **17. Understanding and Enhancing Robustness of Concept-based Models**

cs.LG

Accepted at AAAI 2023. Extended Version

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16080v1) [paper-pdf](http://arxiv.org/pdf/2211.16080v1)

**Authors**: Sanchit Sinha, Mengdi Huai, Jianhui Sun, Aidong Zhang

**Abstract**: Rising usage of deep neural networks to perform decision making in critical applications like medical diagnosis and financial analysis have raised concerns regarding their reliability and trustworthiness. As automated systems become more mainstream, it is important their decisions be transparent, reliable and understandable by humans for better trust and confidence. To this effect, concept-based models such as Concept Bottleneck Models (CBMs) and Self-Explaining Neural Networks (SENN) have been proposed which constrain the latent space of a model to represent high level concepts easily understood by domain experts in the field. Although concept-based models promise a good approach to both increasing explainability and reliability, it is yet to be shown if they demonstrate robustness and output consistent concepts under systematic perturbations to their inputs. To better understand performance of concept-based models on curated malicious samples, in this paper, we aim to study their robustness to adversarial perturbations, which are also known as the imperceptible changes to the input data that are crafted by an attacker to fool a well-learned concept-based model. Specifically, we first propose and analyze different malicious attacks to evaluate the security vulnerability of concept based models. Subsequently, we propose a potential general adversarial training-based defense mechanism to increase robustness of these systems to the proposed malicious attacks. Extensive experiments on one synthetic and two real-world datasets demonstrate the effectiveness of the proposed attacks and the defense approach.



## **18. Model Extraction Attack against Self-supervised Speech Models**

cs.SD

Submitted to ICASSP 2023

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16044v1) [paper-pdf](http://arxiv.org/pdf/2211.16044v1)

**Authors**: Tsu-Yuan Hsu, Chen-An Li, Tung-Yu Wu, Hung-yi Lee

**Abstract**: Self-supervised learning (SSL) speech models generate meaningful representations of given clips and achieve incredible performance across various downstream tasks. Model extraction attack (MEA) often refers to an adversary stealing the functionality of the victim model with only query access. In this work, we study the MEA problem against SSL speech model with a small number of queries. We propose a two-stage framework to extract the model. In the first stage, SSL is conducted on the large-scale unlabeled corpus to pre-train a small speech model. Secondly, we actively sample a small portion of clips from the unlabeled corpus and query the target model with these clips to acquire their representations as labels for the small model's second-stage training. Experiment results show that our sampling methods can effectively extract the target model without knowing any information about its model architecture.



## **19. AdvMask: A Sparse Adversarial Attack Based Data Augmentation Method for Image Classification**

cs.CV

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.16040v1) [paper-pdf](http://arxiv.org/pdf/2211.16040v1)

**Authors**: Suorong Yang, Jinqiao Li, Jian Zhao, Furao Shen

**Abstract**: Data augmentation is a widely used technique for enhancing the generalization ability of convolutional neural networks (CNNs) in image classification tasks. Occlusion is a critical factor that affects on the generalization ability of image classification models. In order to generate new samples, existing data augmentation methods based on information deletion simulate occluded samples by randomly removing some areas in the images. However, those methods cannot delete areas of the images according to their structural features of the images. To solve those problems, we propose a novel data augmentation method, AdvMask, for image classification tasks. Instead of randomly removing areas in the images, AdvMask obtains the key points that have the greatest influence on the classification results via an end-to-end sparse adversarial attack module. Therefore, we can find the most sensitive points of the classification results without considering the diversity of various image appearance and shapes of the object of interest. In addition, a data augmentation module is employed to generate structured masks based on the key points, thus forcing the CNN classification models to seek other relevant content when the most discriminative content is hidden. AdvMask can effectively improve the performance of classification models in the testing process. The experimental results on various datasets and CNN models verify that the proposed method outperforms other previous data augmentation methods in image classification tasks.



## **20. Interpretations Cannot Be Trusted: Stealthy and Effective Adversarial Perturbations against Interpretable Deep Learning**

cs.CR

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15926v1) [paper-pdf](http://arxiv.org/pdf/2211.15926v1)

**Authors**: Eldor Abdukhamidov, Mohammed Abuhamad, Simon S. Woo, Eric Chan-Tin, Tamer Abuhmed

**Abstract**: Deep learning methods have gained increased attention in various applications due to their outstanding performance. For exploring how this high performance relates to the proper use of data artifacts and the accurate problem formulation of a given task, interpretation models have become a crucial component in developing deep learning-based systems. Interpretation models enable the understanding of the inner workings of deep learning models and offer a sense of security in detecting the misuse of artifacts in the input data. Similar to prediction models, interpretation models are also susceptible to adversarial inputs. This work introduces two attacks, AdvEdge and AdvEdge$^{+}$, that deceive both the target deep learning model and the coupled interpretation model. We assess the effectiveness of proposed attacks against two deep learning model architectures coupled with four interpretation models that represent different categories of interpretation models. Our experiments include the attack implementation using various attack frameworks. We also explore the potential countermeasures against such attacks. Our analysis shows the effectiveness of our attacks in terms of deceiving the deep learning models and their interpreters, and highlights insights to improve and circumvent the attacks.



## **21. Training Time Adversarial Attack Aiming the Vulnerability of Continual Learning**

cs.LG

Accepted at NeurIPS 2022 ML Safety Workshop

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15875v1) [paper-pdf](http://arxiv.org/pdf/2211.15875v1)

**Authors**: Gyojin Han, Jaehyun Choi, Hyeong Gwon Hong, Junmo Kim

**Abstract**: Generally, regularization-based continual learning models limit access to the previous task data to imitate the real-world setting which has memory and privacy issues. However, this introduces a problem in these models by not being able to track the performance on each task. In other words, current continual learning methods are vulnerable to attacks done on the previous task. We demonstrate the vulnerability of regularization-based continual learning methods by presenting simple task-specific training time adversarial attack that can be used in the learning process of a new task. Training data generated by the proposed attack causes performance degradation on a specific task targeted by the attacker. Experiment results justify the vulnerability proposed in this paper and demonstrate the importance of developing continual learning models that are robust to adversarial attack.



## **22. How Important are Good Method Names in Neural Code Generation? A Model Robustness Perspective**

cs.SE

UNDER REVIEW

**SubmitDate**: 2022-11-29    [abs](http://arxiv.org/abs/2211.15844v1) [paper-pdf](http://arxiv.org/pdf/2211.15844v1)

**Authors**: Guang Yang, Yu Zhou, Wenhua Yang, Tao Yue, Xiang Chen, Taolue Chen

**Abstract**: Pre-trained code generation models (PCGMs) have been widely applied in neural code generation which can generate executable code from functional descriptions in natural languages, possibly together with signatures. Despite substantial performance improvement of PCGMs, the role of method names in neural code generation has not been thoroughly investigated. In this paper, we study and demonstrate the potential of benefiting from method names to enhance the performance of PCGMs, from a model robustness perspective. Specifically, we propose a novel approach, named RADAR (neuRAl coDe generAtor Robustifier). RADAR consists of two components: RADAR-Attack and RADAR-Defense. The former attacks a PCGM by generating adversarial method names as part of the input, which are semantic and visual similar to the original input, but may trick the PCGM to generate completely unrelated code snippets. As a countermeasure to such attacks, RADAR-Defense synthesizes a new method name from the functional description and supplies it to the PCGM. Evaluation results show that RADAR-Attack can, e.g., reduce the CodeBLEU of generated code by 19.72% to 38.74% in three state-of-the-art PCGMs (i.e., CodeGPT, PLBART, and CodeT5). Moreover, RADAR-Defense is able to reinstate the performance of PCGMs with synthesized method names. These results highlight the importance of good method names in neural code generation and implicate the benefits of studying model robustness in software engineering.



## **23. Attack on Unfair ToS Clause Detection: A Case Study using Universal Adversarial Triggers**

cs.CL

Accepted at NLLP@EMNLP2022

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15556v1) [paper-pdf](http://arxiv.org/pdf/2211.15556v1)

**Authors**: Shanshan Xu, Irina Broda, Rashid Haddad, Marco Negrini, Matthias Grabmair

**Abstract**: Recent work has demonstrated that natural language processing techniques can support consumer protection by automatically detecting unfair clauses in the Terms of Service (ToS) Agreement. This work demonstrates that transformer-based ToS analysis systems are vulnerable to adversarial attacks. We conduct experiments attacking an unfair-clause detector with universal adversarial triggers. Experiments show that a minor perturbation of the text can considerably reduce the detection performance. Moreover, to measure the detectability of the triggers, we conduct a detailed human evaluation study by collecting both answer accuracy and response time from the participants. The results show that the naturalness of the triggers remains key to tricking readers.



## **24. Adversarial Detection by Approximation of Ensemble Boundary**

cs.LG

6 pages, 3 figures, 5 tables

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.10227v2) [paper-pdf](http://arxiv.org/pdf/2211.10227v2)

**Authors**: T. Windeatt

**Abstract**: A spectral approximation of a Boolean function is proposed for approximating the decision boundary of an ensemble of Deep Neural Networks (DNNs) solving two-class pattern recognition problems. The Walsh combination of relatively weak DNN classifiers is shown experimentally to be capable of detecting adversarial attacks. By observing the difference in Walsh coefficient approximation between clean and adversarial images, it appears that transferability of attack may be used for detection. Approximating the decision boundary may also aid in understanding the learning and transferability properties of DNNs. While the experiments here use images, the proposed approach of modelling two-class ensemble decision boundaries could in principle be applied to any application area.



## **25. DeepSE-WF: Unified Security Estimation for Website Fingerprinting Defenses**

cs.CR

Major revision - added experiments with new dataset and alternative  neural network architectures for estimating the BER

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2203.04428v2) [paper-pdf](http://arxiv.org/pdf/2203.04428v2)

**Authors**: Alexander Veicht, Cedric Renggli, Diogo Barradas

**Abstract**: Website fingerprinting (WF) attacks, usually conducted with the help of a machine learning-based classifier, enable a network eavesdropper to pinpoint which web page a user is accessing through the inspection of traffic patterns. These attacks have been shown to succeed even when users browse the Internet through encrypted tunnels, e.g., through Tor or VPNs. To assess the security of new defenses against WF attacks, recent works have proposed feature-dependent theoretical frameworks that estimate the Bayes error of an adversary's features set or the mutual information leaked by manually-crafted features. Unfortunately, as state-of-the-art WF attacks increasingly rely on deep learning and latent feature spaces, security estimations based on simpler (and less informative) manually-crafted features can no longer be trusted to assess the potential success of a WF adversary in defeating such defenses. In this work, we propose DeepSE-WF, a novel WF security estimation framework that leverages specialized kNN-based estimators to produce Bayes error and mutual information estimates from learned latent feature spaces, thus bridging the gap between current WF attacks and security estimation methods. Our evaluation reveals that DeepSE-WF produces tighter security estimates than previous frameworks, reducing the required computational resources to output security estimations by one order of magnitude.



## **26. Security Analysis of the Consumer Remote SIM Provisioning Protocol**

cs.CR

33 pages, 8 figures, Associated ProVerif model files located at  https://github.com/peltona/rsp_model

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15323v1) [paper-pdf](http://arxiv.org/pdf/2211.15323v1)

**Authors**: Abu Shohel Ahmed, Aleksi Peltonen, Mohit Sethi, Tuomas Aura

**Abstract**: Remote SIM provisioning (RSP) for consumer devices is the protocol specified by the GSM Association for downloading SIM profiles into a secure element in a mobile device. The process is commonly known as eSIM, and it is expected to replace removable SIM cards. The security of the protocol is critical because the profile includes the credentials with which the mobile device will authenticate to the mobile network. In this paper, we present a formal security analysis of the consumer RSP protocol. We model the multi-party protocol in applied pi calculus, define formal security goals, and verify them in ProVerif. The analysis shows that the consumer RSP protocol protects against a network adversary when all the intended participants are honest. However, we also model the protocol in realistic partial compromise scenarios where the adversary controls a legitimate participant or communication channel. The security failures in the partial compromise scenarios reveal weaknesses in the protocol design. The most important observation is that the security of RSP depends unnecessarily on it being encapsulated in a TLS tunnel. Also, the lack of pre-established identifiers means that a compromised download server anywhere in the world or a compromised secure element can be used for attacks against RSP between honest participants. Additionally, the lack of reliable methods for verifying user intent can lead to serious security failures. Based on the findings, we recommend practical improvements to RSP implementations, to future versions of the specification, and to mobile operator processes to increase the robustness of eSIM security.



## **27. Adversarial Artifact Detection in EEG-Based Brain-Computer Interfaces**

cs.CR

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2212.00727v1) [paper-pdf](http://arxiv.org/pdf/2212.00727v1)

**Authors**: Xiaoqing Chen, Dongrui Wu

**Abstract**: Machine learning has achieved great success in electroencephalogram (EEG) based brain-computer interfaces (BCIs). Most existing BCI research focused on improving its accuracy, but few had considered its security. Recent studies, however, have shown that EEG-based BCIs are vulnerable to adversarial attacks, where small perturbations added to the input can cause misclassification. Detection of adversarial examples is crucial to both the understanding of this phenomenon and the defense. This paper, for the first time, explores adversarial detection in EEG-based BCIs. Experiments on two EEG datasets using three convolutional neural networks were performed to verify the performances of multiple detection approaches. We showed that both white-box and black-box attacks can be detected, and the former are easier to detect.



## **28. Rethinking the Number of Shots in Robust Model-Agnostic Meta-Learning**

cs.CV

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15180v1) [paper-pdf](http://arxiv.org/pdf/2211.15180v1)

**Authors**: Xiaoyue Duan, Guoliang Kang, Runqi Wang, Shumin Han, Song Xue, Tian Wang, Baochang Zhang

**Abstract**: Robust Model-Agnostic Meta-Learning (MAML) is usually adopted to train a meta-model which may fast adapt to novel classes with only a few exemplars and meanwhile remain robust to adversarial attacks. The conventional solution for robust MAML is to introduce robustness-promoting regularization during meta-training stage. With such a regularization, previous robust MAML methods simply follow the typical MAML practice that the number of training shots should match with the number of test shots to achieve an optimal adaptation performance. However, although the robustness can be largely improved, previous methods sacrifice clean accuracy a lot. In this paper, we observe that introducing robustness-promoting regularization into MAML reduces the intrinsic dimension of clean sample features, which results in a lower capacity of clean representations. This may explain why the clean accuracy of previous robust MAML methods drops severely. Based on this observation, we propose a simple strategy, i.e., increasing the number of training shots, to mitigate the loss of intrinsic dimension caused by robustness-promoting regularization. Though simple, our method remarkably improves the clean accuracy of MAML without much loss of robustness, producing a robust yet accurate model. Extensive experiments demonstrate that our method outperforms prior arts in achieving a better trade-off between accuracy and robustness. Besides, we observe that our method is less sensitive to the number of fine-tuning steps during meta-training, which allows for a reduced number of fine-tuning steps to improve training efficiency.



## **29. Adversarial Attack on Radar-based Environment Perception Systems**

cs.CR

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.01112v2) [paper-pdf](http://arxiv.org/pdf/2211.01112v2)

**Authors**: Amira Guesmi, Ihsen Alouani

**Abstract**: Due to their robustness to degraded capturing conditions, radars are widely used for environment perception, which is a critical task in applications like autonomous vehicles. More specifically, Ultra-Wide Band (UWB) radars are particularly efficient for short range settings as they carry rich information on the environment. Recent UWB-based systems rely on Machine Learning (ML) to exploit the rich signature of these sensors. However, ML classifiers are susceptible to adversarial examples, which are created from raw data to fool the classifier such that it assigns the input to the wrong class. These attacks represent a serious threat to systems integrity, especially for safety-critical applications. In this work, we present a new adversarial attack on UWB radars in which an adversary injects adversarial radio noise in the wireless channel to cause an obstacle recognition failure. First, based on signals collected in real-life environment, we show that conventional attacks fail to generate robust noise under realistic conditions. We propose a-RNA, i.e., Adversarial Radio Noise Attack to overcome these issues. Specifically, a-RNA generates an adversarial noise that is efficient without synchronization between the input signal and the noise. Moreover, a-RNA generated noise is, by-design, robust against pre-processing countermeasures such as filtering-based defenses. Moreover, in addition to the undetectability objective by limiting the noise magnitude budget, a-RNA is also efficient in the presence of sophisticated defenses in the spectral domain by introducing a frequency budget. We believe this work should alert about potentially critical implementations of adversarial attacks on radar systems that should be taken seriously.



## **30. Imperceptible Adversarial Attack via Invertible Neural Networks**

cs.CV

**SubmitDate**: 2022-11-28    [abs](http://arxiv.org/abs/2211.15030v1) [paper-pdf](http://arxiv.org/pdf/2211.15030v1)

**Authors**: Zihan Chen, Ziyue Wang, Junjie Huang, Wentao Zhao, Xiao Liu, Dejian Guan

**Abstract**: Adding perturbations via utilizing auxiliary gradient information or discarding existing details of the benign images are two common approaches for generating adversarial examples. Though visual imperceptibility is the desired property of adversarial examples, conventional adversarial attacks still generate traceable adversarial perturbations. In this paper, we introduce a novel Adversarial Attack via Invertible Neural Networks (AdvINN) method to produce robust and imperceptible adversarial examples. Specifically, AdvINN fully takes advantage of the information preservation property of Invertible Neural Networks and thereby generates adversarial examples by simultaneously adding class-specific semantic information of the target class and dropping discriminant information of the original class. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet-1K demonstrate that the proposed AdvINN method can produce less imperceptible adversarial images than the state-of-the-art methods and AdvINN yields more robust adversarial examples with high confidence compared to other adversarial attacks.



## **31. Adversarial Rademacher Complexity of Deep Neural Networks**

cs.LG

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14966v1) [paper-pdf](http://arxiv.org/pdf/2211.14966v1)

**Authors**: Jiancong Xiao, Yanbo Fan, Ruoyu Sun, Zhi-Quan Luo

**Abstract**: Deep neural networks are vulnerable to adversarial attacks. Ideally, a robust model shall perform well on both the perturbed training data and the unseen perturbed test data. It is found empirically that fitting perturbed training data is not hard, but generalizing to perturbed test data is quite difficult. To better understand adversarial generalization, it is of great interest to study the adversarial Rademacher complexity (ARC) of deep neural networks. However, how to bound ARC in multi-layers cases is largely unclear due to the difficulty of analyzing adversarial loss in the definition of ARC. There have been two types of attempts of ARC. One is to provide the upper bound of ARC in linear and one-hidden layer cases. However, these approaches seem hard to extend to multi-layer cases. Another is to modify the adversarial loss and provide upper bounds of Rademacher complexity on such surrogate loss in multi-layer cases. However, such variants of Rademacher complexity are not guaranteed to be bounds for meaningful robust generalization gaps (RGG). In this paper, we provide a solution to this unsolved problem. Specifically, we provide the first bound of adversarial Rademacher complexity of deep neural networks. Our approach is based on covering numbers. We provide a method to handle the robustify function classes of DNNs such that we can calculate the covering numbers. Finally, we provide experiments to study the empirical implication of our bounds and provide an analysis of poor adversarial generalization.



## **32. Foiling Explanations in Deep Neural Networks**

cs.CV

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14860v1) [paper-pdf](http://arxiv.org/pdf/2211.14860v1)

**Authors**: Snir Vitrack Tamam, Raz Lapid, Moshe Sipper

**Abstract**: Deep neural networks (DNNs) have greatly impacted numerous fields over the past decade. Yet despite exhibiting superb performance over many problems, their black-box nature still poses a significant challenge with respect to explainability. Indeed, explainable artificial intelligence (XAI) is crucial in several fields, wherein the answer alone -- sans a reasoning of how said answer was derived -- is of little value. This paper uncovers a troubling property of explanation methods for image-based DNNs: by making small visual changes to the input image -- hardly influencing the network's output -- we demonstrate how explanations may be arbitrarily manipulated through the use of evolution strategies. Our novel algorithm, AttaXAI, a model-agnostic, adversarial attack on XAI algorithms, only requires access to the output logits of a classifier and to the explanation map; these weak assumptions render our approach highly useful where real-world models and data are concerned. We compare our method's performance on two benchmark datasets -- CIFAR100 and ImageNet -- using four different pretrained deep-learning models: VGG16-CIFAR100, VGG16-ImageNet, MobileNet-CIFAR100, and Inception-v3-ImageNet. We find that the XAI methods can be manipulated without the use of gradients or other model internals. Our novel algorithm is successfully able to manipulate an image in a manner imperceptible to the human eye, such that the XAI method outputs a specific explanation map. To our knowledge, this is the first such method in a black-box setting, and we believe it has significant value where explainability is desired, required, or legally mandatory.



## **33. Traditional Classification Neural Networks are Good Generators: They are Competitive with DDPMs and GANs**

cs.CV

This paper has 29 pages with 22 figures, including rich supplementary  information

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14794v1) [paper-pdf](http://arxiv.org/pdf/2211.14794v1)

**Authors**: Guangrun Wang, Philip H. S. Torr

**Abstract**: Classifiers and generators have long been separated. We break down this separation and showcase that conventional neural network classifiers can generate high-quality images of a large number of categories, being comparable to the state-of-the-art generative models (e.g., DDPMs and GANs). We achieve this by computing the partial derivative of the classification loss function with respect to the input to optimize the input to produce an image. Since it is widely known that directly optimizing the inputs is similar to targeted adversarial attacks incapable of generating human-meaningful images, we propose a mask-based stochastic reconstruction module to make the gradients semantic-aware to synthesize plausible images. We further propose a progressive-resolution technique to guarantee fidelity, which produces photorealistic images. Furthermore, we introduce a distance metric loss and a non-trivial distribution loss to ensure classification neural networks can synthesize diverse and high-fidelity images. Using traditional neural network classifiers, we can generate good-quality images of 256$\times$256 resolution on ImageNet. Intriguingly, our method is also applicable to text-to-image generation by regarding image-text foundation models as generalized classifiers.   Proving that classifiers have learned the data distribution and are ready for image generation has far-reaching implications, for classifiers are much easier to train than generative models like DDPMs and GANs. We don't even need to train classification models because tons of public ones are available for download. Also, this holds great potential for the interpretability and robustness of classifiers.



## **34. Plausible Adversarial Attacks on Direct Parameter Inference Models in Astrophysics**

astro-ph.CO

Accepted submission to Machine Learning and the Physical Sciences  workshop, NeurIPS 2022

**SubmitDate**: 2022-11-27    [abs](http://arxiv.org/abs/2211.14788v1) [paper-pdf](http://arxiv.org/pdf/2211.14788v1)

**Authors**: Benjamin Horowitz, Peter Melchior

**Abstract**: In this abstract we explore the possibility of introducing biases in physical parameter inference models from adversarial-type attacks. In particular, we inject small amplitude systematics into inputs to a mixture density networks tasked with inferring cosmological parameters from observed data. The systematics are constructed analogously to white-box adversarial attacks. We find that the analysis network can be tricked into spurious detection of new physics in cases where standard cosmological estimators would be insensitive. This calls into question the robustness of such networks and their utility for reliably detecting new physics.



## **35. Game Theoretic Mixed Experts for Combinational Adversarial Machine Learning**

cs.LG

21 pages, 6 figures

**SubmitDate**: 2022-11-26    [abs](http://arxiv.org/abs/2211.14669v1) [paper-pdf](http://arxiv.org/pdf/2211.14669v1)

**Authors**: Ethan Rathbun, Kaleel Mahmood, Sohaib Ahmad, Caiwen Ding, Marten van Dijk

**Abstract**: Recent advances in adversarial machine learning have shown that defenses considered to be robust are actually susceptible to adversarial attacks which are specifically tailored to target their weaknesses. These defenses include Barrage of Random Transforms (BaRT), Friendly Adversarial Training (FAT), Trash is Treasure (TiT) and ensemble models made up of Vision Transformers (ViTs), Big Transfer models and Spiking Neural Networks (SNNs). A natural question arises: how can one best leverage a combination of adversarial defenses to thwart such attacks? In this paper, we provide a game-theoretic framework for ensemble adversarial attacks and defenses which answers this question. In addition to our framework we produce the first adversarial defense transferability study to further motivate a need for combinational defenses utilizing a diverse set of defense architectures. Our framework is called Game theoretic Mixed Experts (GaME) and is designed to find the Mixed-Nash strategy for a defender when facing an attacker employing compositional adversarial attacks. We show that this framework creates an ensemble of defenses with greater robustness than multiple state-of-the-art, single-model defenses in addition to combinational defenses with uniform probability distributions. Overall, our framework and analyses advance the field of adversarial machine learning by yielding new insights into compositional attack and defense formulations.



## **36. Minimax Problems with Coupled Linear Constraints: Computational Complexity, Duality and Solution Methods**

math.OC

**SubmitDate**: 2022-11-26    [abs](http://arxiv.org/abs/2110.11210v2) [paper-pdf](http://arxiv.org/pdf/2110.11210v2)

**Authors**: Ioannis Tsaknakis, Mingyi Hong, Shuzhong Zhang

**Abstract**: In this work we study a special minimax problem where there are linear constraints that couple both the minimization and maximization decision variables. The problem is a generalization of the traditional saddle point problem (which does not have the coupling constraint), and it finds applications in wireless communication, game theory, transportation, just to name a few. We show that the considered problem is challenging, in the sense that it violates the classical max-min inequality, and that it is NP-hard even under very strong assumptions (e.g., when the objective is strongly convex-strongly concave). We then develop a duality theory for it, and analyze conditions under which the duality gap becomes zero. Finally, we study a class of stationary solutions defined based on the dual problem, and evaluate their practical performance in an application on adversarial attacks on network flow problems.



## **37. SegPGD: An Effective and Efficient Adversarial Attack for Evaluating and Boosting Segmentation Robustness**

cs.CV

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2207.12391v2) [paper-pdf](http://arxiv.org/pdf/2207.12391v2)

**Authors**: Jindong Gu, Hengshuang Zhao, Volker Tresp, Philip Torr

**Abstract**: Deep neural network-based image classifications are vulnerable to adversarial perturbations. The image classifications can be easily fooled by adding artificial small and imperceptible perturbations to input images. As one of the most effective defense strategies, adversarial training was proposed to address the vulnerability of classification models, where the adversarial examples are created and injected into training data during training. The attack and defense of classification models have been intensively studied in past years. Semantic segmentation, as an extension of classifications, has also received great attention recently. Recent work shows a large number of attack iterations are required to create effective adversarial examples to fool segmentation models. The observation makes both robustness evaluation and adversarial training on segmentation models challenging. In this work, we propose an effective and efficient segmentation attack method, dubbed SegPGD. Besides, we provide a convergence analysis to show the proposed SegPGD can create more effective adversarial examples than PGD under the same number of attack iterations. Furthermore, we propose to apply our SegPGD as the underlying attack method for segmentation adversarial training. Since SegPGD can create more effective adversarial examples, the adversarial training with our SegPGD can boost the robustness of segmentation models. Our proposals are also verified with experiments on popular Segmentation model architectures and standard segmentation datasets.



## **38. Beyond Smoothing: Unsupervised Graph Representation Learning with Edge Heterophily Discriminating**

cs.LG

14 pages, 7 tables, 6 figures, accepted by AAAI 2023

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.14065v1) [paper-pdf](http://arxiv.org/pdf/2211.14065v1)

**Authors**: Yixin Liu, Yizhen Zheng, Daokun Zhang, Vincent CS Lee, Shirui Pan

**Abstract**: Unsupervised graph representation learning (UGRL) has drawn increasing research attention and achieved promising results in several graph analytic tasks. Relying on the homophily assumption, existing UGRL methods tend to smooth the learned node representations along all edges, ignoring the existence of heterophilic edges that connect nodes with distinct attributes. As a result, current methods are hard to generalize to heterophilic graphs where dissimilar nodes are widely connected, and also vulnerable to adversarial attacks. To address this issue, we propose a novel unsupervised Graph Representation learning method with Edge hEterophily discriminaTing (GREET) which learns representations by discriminating and leveraging homophilic edges and heterophilic edges. To distinguish two types of edges, we build an edge discriminator that infers edge homophily/heterophily from feature and structure information. We train the edge discriminator in an unsupervised way through minimizing the crafted pivot-anchored ranking loss, with randomly sampled node pairs acting as pivots. Node representations are learned through contrasting the dual-channel encodings obtained from the discriminated homophilic and heterophilic edges. With an effective interplaying scheme, edge discriminating and representation learning can mutually boost each other during the training phase. We conducted extensive experiments on 14 benchmark datasets and multiple learning scenarios to demonstrate the superiority of GREET.



## **39. Cross-Domain Ensemble Distillation for Domain Generalization**

cs.CV

Accepted to ECCV 2022. Code is available at  http://github.com/leekyungmoon/XDED

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.14058v1) [paper-pdf](http://arxiv.org/pdf/2211.14058v1)

**Authors**: Kyungmoon Lee, Sungyeon Kim, Suha Kwak

**Abstract**: Domain generalization is the task of learning models that generalize to unseen target domains. We propose a simple yet effective method for domain generalization, named cross-domain ensemble distillation (XDED), that learns domain-invariant features while encouraging the model to converge to flat minima, which recently turned out to be a sufficient condition for domain generalization. To this end, our method generates an ensemble of the output logits from training data with the same label but from different domains and then penalizes each output for the mismatch with the ensemble. Also, we present a de-stylization technique that standardizes features to encourage the model to produce style-consistent predictions even in an arbitrary target domain. Our method greatly improves generalization capability in public benchmarks for cross-domain image classification, cross-dataset person re-ID, and cross-dataset semantic segmentation. Moreover, we show that models learned by our method are robust against adversarial attacks and image corruptions.



## **40. Cross-Quality LFW: A Database for Analyzing Cross-Resolution Image Face Recognition in Unconstrained Environments**

cs.CV

9 pages, 4 figures, 2 tables

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2108.10290v3) [paper-pdf](http://arxiv.org/pdf/2108.10290v3)

**Authors**: Martin Knoche, Stefan Hörmann, Gerhard Rigoll

**Abstract**: Real-world face recognition applications often deal with suboptimal image quality or resolution due to different capturing conditions such as various subject-to-camera distances, poor camera settings, or motion blur. This characteristic has an unignorable effect on performance. Recent cross-resolution face recognition approaches used simple, arbitrary, and unrealistic down- and up-scaling techniques to measure robustness against real-world edge-cases in image quality. Thus, we propose a new standardized benchmark dataset and evaluation protocol derived from the famous Labeled Faces in the Wild (LFW). In contrast to previous derivatives, which focus on pose, age, similarity, and adversarial attacks, our Cross-Quality Labeled Faces in the Wild (XQLFW) maximizes the quality difference. It contains only more realistic synthetically degraded images when necessary. Our proposed dataset is then used to further investigate the influence of image quality on several state-of-the-art approaches. With XQLFW, we show that these models perform differently in cross-quality cases, and hence, the generalizing capability is not accurately predicted by their performance on LFW. Additionally, we report baseline accuracy with recent deep learning models explicitly trained for cross-resolution applications and evaluate the susceptibility to image quality. To encourage further research in cross-resolution face recognition and incite the assessment of image quality robustness, we publish the database and code for evaluation.



## **41. Let Graph be the Go Board: Gradient-free Node Injection Attack for Graph Neural Networks via Reinforcement Learning**

cs.LG

AAAI 2023. v2: update acknowledgement section. arXiv admin note:  substantial text overlap with arXiv:2202.09389

**SubmitDate**: 2022-11-25    [abs](http://arxiv.org/abs/2211.10782v2) [paper-pdf](http://arxiv.org/pdf/2211.10782v2)

**Authors**: Mingxuan Ju, Yujie Fan, Chuxu Zhang, Yanfang Ye

**Abstract**: Graph Neural Networks (GNNs) have drawn significant attentions over the years and been broadly applied to essential applications requiring solid robustness or vigorous security standards, such as product recommendation and user behavior modeling. Under these scenarios, exploiting GNN's vulnerabilities and further downgrading its performance become extremely incentive for adversaries. Previous attackers mainly focus on structural perturbations or node injections to the existing graphs, guided by gradients from the surrogate models. Although they deliver promising results, several limitations still exist. For the structural perturbation attack, to launch a proposed attack, adversaries need to manipulate the existing graph topology, which is impractical in most circumstances. Whereas for the node injection attack, though being more practical, current approaches require training surrogate models to simulate a white-box setting, which results in significant performance downgrade when the surrogate architecture diverges from the actual victim model. To bridge these gaps, in this paper, we study the problem of black-box node injection attack, without training a potentially misleading surrogate model. Specifically, we model the node injection attack as a Markov decision process and propose Gradient-free Graph Advantage Actor Critic, namely G2A2C, a reinforcement learning framework in the fashion of advantage actor critic. By directly querying the victim model, G2A2C learns to inject highly malicious nodes with extremely limited attacking budgets, while maintaining a similar node feature distribution. Through our comprehensive experiments over eight acknowledged benchmark datasets with different characteristics, we demonstrate the superior performance of our proposed G2A2C over the existing state-of-the-art attackers. Source code is publicly available at: https://github.com/jumxglhf/G2A2C}.



## **42. SAGA: Spectral Adversarial Geometric Attack on 3D Meshes**

cs.CV

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13775v1) [paper-pdf](http://arxiv.org/pdf/2211.13775v1)

**Authors**: Tomer Stolik, Itai Lang, Shai Avidan

**Abstract**: A triangular mesh is one of the most popular 3D data representations. As such, the deployment of deep neural networks for mesh processing is widely spread and is increasingly attracting more attention. However, neural networks are prone to adversarial attacks, where carefully crafted inputs impair the model's functionality. The need to explore these vulnerabilities is a fundamental factor in the future development of 3D-based applications. Recently, mesh attacks were studied on the semantic level, where classifiers are misled to produce wrong predictions. Nevertheless, mesh surfaces possess complex geometric attributes beyond their semantic meaning, and their analysis often includes the need to encode and reconstruct the geometry of the shape.   We propose a novel framework for a geometric adversarial attack on a 3D mesh autoencoder. In this setting, an adversarial input mesh deceives the autoencoder by forcing it to reconstruct a different geometric shape at its output. The malicious input is produced by perturbing a clean shape in the spectral domain. Our method leverages the spectral decomposition of the mesh along with additional mesh-related properties to obtain visually credible results that consider the delicacy of surface distortions. Our code is publicly available at https://github.com/StolikTomer/SAGA.



## **43. Backdoor Attack and Defense in Federated Generative Adversarial Network-based Medical Image Synthesis**

cs.CV

25 pages, 7 figures. arXiv admin note: text overlap with  arXiv:2207.00762

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2210.10886v2) [paper-pdf](http://arxiv.org/pdf/2210.10886v2)

**Authors**: Ruinan Jin, Xiaoxiao Li

**Abstract**: Deep Learning-based image synthesis techniques have been applied in healthcare research for generating medical images to support open research and augment medical datasets. Training generative adversarial neural networks (GANs) usually require large amounts of training data. Federated learning (FL) provides a way of training a central model using distributed data while keeping raw data locally. However, given that the FL server cannot access the raw data, it is vulnerable to backdoor attacks, an adversarial by poisoning training data. Most backdoor attack strategies focus on classification models and centralized domains. It is still an open question if the existing backdoor attacks can affect GAN training and, if so, how to defend against the attack in the FL setting. In this work, we investigate the overlooked issue of backdoor attacks in federated GANs (FedGANs). The success of this attack is subsequently determined to be the result of some local discriminators overfitting the poisoned data and corrupting the local GAN equilibrium, which then further contaminates other clients when averaging the generator's parameters and yields high generator loss. Therefore, we proposed FedDetect, an efficient and effective way of defending against the backdoor attack in the FL setting, which allows the server to detect the client's adversarial behavior based on their losses and block the malicious clients. Our extensive experiments on two medical datasets with different modalities demonstrate the backdoor attack on FedGANs can result in synthetic images with low fidelity. After detecting and suppressing the detected malicious clients using the proposed defense strategy, we show that FedGANs can synthesize high-quality medical datasets (with labels) for data augmentation to improve classification models' performance.



## **44. Enhancing Targeted Attack Transferability via Diversified Weight Pruning**

cs.CV

8 pages + Appendix

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2208.08677v2) [paper-pdf](http://arxiv.org/pdf/2208.08677v2)

**Authors**: Hung-Jui Wang, Yu-Yu Wu, Shang-Tse Chen

**Abstract**: Malicious attackers can generate targeted adversarial examples by imposing tiny noises, forcing neural networks to produce specific incorrect outputs. With cross-model transferability, network models remain vulnerable even in black-box settings. Recent studies have shown the effectiveness of ensemble-based methods in generating transferable adversarial examples. To further enhance transferability, model augmentation methods aim to produce more networks participating in the ensemble. However, existing model augmentation methods are only proven effective in untargeted attacks. In this work, we propose Diversified Weight Pruning (DWP), a novel model augmentation technique for generating transferable targeted attacks. DWP leverages the weight pruning method commonly used in model compression. Compared with prior work, DWP protects necessary connections and ensures the diversity of the pruned models simultaneously, which we show are crucial for targeted transferability. Experiments on the ImageNet-compatible dataset under various and more challenging scenarios confirm the effectiveness: transferring to adversarially trained models, Non-CNN architectures, and Google Cloud Vision. The results show that our proposed DWP improves the targeted attack success rates with up to $10.1$%, $6.6$%, and $7.0$% on the combination of state-of-the-art methods, respectively. The source code will be made available after acceptance.



## **45. Tracking Dataset IP Use in Deep Neural Networks**

cs.CR

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13535v1) [paper-pdf](http://arxiv.org/pdf/2211.13535v1)

**Authors**: Seonhye Park, Alsharif Abuadbba, Shuo Wang, Kristen Moore, Yansong Gao, Hyoungshick Kim, Surya Nepal

**Abstract**: Training highly performant deep neural networks (DNNs) typically requires the collection of a massive dataset and the use of powerful computing resources. Therefore, unauthorized redistribution of private pre-trained DNNs may cause severe economic loss for model owners. For protecting the ownership of DNN models, DNN watermarking schemes have been proposed by embedding secret information in a DNN model and verifying its presence for model ownership. However, existing DNN watermarking schemes compromise the model utility and are vulnerable to watermark removal attacks because a model is modified with a watermark. Alternatively, a new approach dubbed DEEPJUDGE was introduced to measure the similarity between a suspect model and a victim model without modifying the victim model. However, DEEPJUDGE would only be designed to detect the case where a suspect model's architecture is the same as a victim model's. In this work, we propose a novel DNN fingerprinting technique dubbed DEEPTASTER to prevent a new attack scenario in which a victim's data is stolen to build a suspect model. DEEPTASTER can effectively detect such data theft attacks even when a suspect model's architecture differs from a victim model's. To achieve this goal, DEEPTASTER generates a few adversarial images with perturbations, transforms them into the Fourier frequency domain, and uses the transformed images to identify the dataset used in a suspect model. The intuition is that those adversarial images can be used to capture the characteristics of DNNs built on a specific dataset. We evaluated the detection accuracy of DEEPTASTER on three datasets with three model architectures under various attack scenarios, including transfer learning, pruning, fine-tuning, and data augmentation. Overall, DEEPTASTER achieves a balanced accuracy of 94.95%, which is significantly better than 61.11% achieved by DEEPJUDGE in the same settings.



## **46. Reliability and Robustness analysis of Machine Learning based Phishing URL Detectors**

cs.CR

Accepted in Transactions of Dependable and Secure Computing  (SI-Reliability and Robustness in AI-Based Cybersecurity Solutions)

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2005.08454v3) [paper-pdf](http://arxiv.org/pdf/2005.08454v3)

**Authors**: Bushra Sabir, M. Ali Babar, Raj Gaire, Alsharif Abuadbba

**Abstract**: ML-based Phishing URL (MLPU) detectors serve as the first level of defence to protect users and organisations from being victims of phishing attacks. Lately, few studies have launched successful adversarial attacks against specific MLPU detectors raising questions about their practical reliability and usage. Nevertheless, the robustness of these systems has not been extensively investigated. Therefore, the security vulnerabilities of these systems, in general, remain primarily unknown which calls for testing the robustness of these systems. In this article, we have proposed a methodology to investigate the reliability and robustness of 50 representative state-of-the-art MLPU models. Firstly, we have proposed a cost-effective Adversarial URL generator URLBUG that created an Adversarial URL dataset. Subsequently, we reproduced 50 MLPU (traditional ML and Deep learning) systems and recorded their baseline performance. Lastly, we tested the considered MLPU systems on Adversarial Dataset and analyzed their robustness and reliability using box plots and heat maps. Our results showed that the generated adversarial URLs have valid syntax and can be registered at a median annual price of \$11.99. Out of 13\% of the already registered adversarial URLs, 63.94\% were used for malicious purposes. Moreover, the considered MLPU models Matthew Correlation Coefficient (MCC) dropped from a median 0.92 to 0.02 when tested against $Adv_\mathrm{data}$, indicating that the baseline MLPU models are unreliable in their current form. Further, our findings identified several security vulnerabilities of these systems and provided future directions for researchers to design dependable and secure MLPU systems.



## **47. Explainable and Safe Reinforcement Learning for Autonomous Air Mobility**

cs.LG

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2211.13474v1) [paper-pdf](http://arxiv.org/pdf/2211.13474v1)

**Authors**: Lei Wang, Hongyu Yang, Yi Lin, Suwan Yin, Yuankai Wu

**Abstract**: Increasing traffic demands, higher levels of automation, and communication enhancements provide novel design opportunities for future air traffic controllers (ATCs). This article presents a novel deep reinforcement learning (DRL) controller to aid conflict resolution for autonomous free flight. Although DRL has achieved important advancements in this field, the existing works pay little attention to the explainability and safety issues related to DRL controllers, particularly the safety under adversarial attacks. To address those two issues, we design a fully explainable DRL framework wherein we: 1) decompose the coupled Q value learning model into a safety-awareness and efficiency (reach the target) one; and 2) use information from surrounding intruders as inputs, eliminating the needs of central controllers. In our simulated experiments, we show that by decoupling the safety-awareness and efficiency, we can exceed performance on free flight control tasks while dramatically improving explainability on practical. In addition, the safety Q learning module provides rich information about the safety situation of environments. To study the safety under adversarial attacks, we additionally propose an adversarial attack strategy that can impose both safety-oriented and efficiency-oriented attacks. The adversarial aims to minimize safety/efficiency by only attacking the agent at a few time steps. In the experiments, our attack strategy increases as many collisions as the uniform attack (i.e., attacking at every time step) by only attacking the agent four times less often, which provide insights into the capabilities and restrictions of the DRL in future ATC designs. The source code is publicly available at https://github.com/WLeiiiii/Gym-ATC-Attack-Project.



## **48. Dikaios: Privacy Auditing of Algorithmic Fairness via Attribute Inference Attacks**

cs.CR

The paper's results and conclusions underwent significant changes.  The updated paper can be found at arXiv:2211.10209

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2202.02242v2) [paper-pdf](http://arxiv.org/pdf/2202.02242v2)

**Authors**: Jan Aalmoes, Vasisht Duddu, Antoine Boutet

**Abstract**: Machine learning (ML) models have been deployed for high-stakes applications. Due to class imbalance in the sensitive attribute observed in the datasets, ML models are unfair on minority subgroups identified by a sensitive attribute, such as race and sex. In-processing fairness algorithms ensure model predictions are independent of sensitive attribute. Furthermore, ML models are vulnerable to attribute inference attacks where an adversary can identify the values of sensitive attribute by exploiting their distinguishable model predictions. Despite privacy and fairness being important pillars of trustworthy ML, the privacy risk introduced by fairness algorithms with respect to attribute leakage has not been studied. We identify attribute inference attacks as an effective measure for auditing blackbox fairness algorithms to enable model builder to account for privacy and fairness in the model design. We proposed Dikaios, a privacy auditing tool for fairness algorithms for model builders which leveraged a new effective attribute inference attack that account for the class imbalance in sensitive attributes through an adaptive prediction threshold. We evaluated Dikaios to perform a privacy audit of two in-processing fairness algorithms over five datasets. We show that our attribute inference attacks with adaptive prediction threshold significantly outperform prior attacks. We highlighted the limitations of in-processing fairness algorithms to ensure indistinguishable predictions across different values of sensitive attributes. Indeed, the attribute privacy risk of these in-processing fairness schemes is highly variable according to the proportion of the sensitive attributes in the dataset. This unpredictable effect of fairness mechanisms on the attribute privacy risk is an important limitation on their utilization which has to be accounted by the model builder.



## **49. Blackbox Attacks via Surrogate Ensemble Search**

cs.LG

Our code is available at https://github.com/CSIPlab/BASES

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2208.03610v2) [paper-pdf](http://arxiv.org/pdf/2208.03610v2)

**Authors**: Zikui Cai, Chengyu Song, Srikanth Krishnamurthy, Amit Roy-Chowdhury, M. Salman Asif

**Abstract**: Blackbox adversarial attacks can be categorized into transfer- and query-based attacks. Transfer methods do not require any feedback from the victim model, but provide lower success rates compared to query-based methods. Query attacks often require a large number of queries for success. To achieve the best of both approaches, recent efforts have tried to combine them, but still require hundreds of queries to achieve high success rates (especially for targeted attacks). In this paper, we propose a novel method for Blackbox Attacks via Surrogate Ensemble Search (BASES) that can generate highly successful blackbox attacks using an extremely small number of queries. We first define a perturbation machine that generates a perturbed image by minimizing a weighted loss function over a fixed set of surrogate models. To generate an attack for a given victim model, we search over the weights in the loss function using queries generated by the perturbation machine. Since the dimension of the search space is small (same as the number of surrogate models), the search requires a small number of queries. We demonstrate that our proposed method achieves better success rate with at least 30x fewer queries compared to state-of-the-art methods on different image classifiers trained with ImageNet. In particular, our method requires as few as 3 queries per image (on average) to achieve more than a 90% success rate for targeted attacks and 1-2 queries per image for over a 99% success rate for untargeted attacks. Our method is also effective on Google Cloud Vision API and achieved a 91% untargeted attack success rate with 2.9 queries per image. We also show that the perturbations generated by our proposed method are highly transferable and can be adopted for hard-label blackbox attacks. We also show effectiveness of BASES for hiding attacks on object detectors.



## **50. Modelling Direct Messaging Networks with Multiple Recipients for Cyber Deception**

cs.CR

**SubmitDate**: 2022-11-24    [abs](http://arxiv.org/abs/2111.11932v2) [paper-pdf](http://arxiv.org/pdf/2111.11932v2)

**Authors**: Kristen Moore, Cody J. Christopher, David Liebowitz, Surya Nepal, Renee Selvey

**Abstract**: Cyber deception is emerging as a promising approach to defending networks and systems against attackers and data thieves. However, despite being relatively cheap to deploy, the generation of realistic content at scale is very costly, due to the fact that rich, interactive deceptive technologies are largely hand-crafted. With recent improvements in Machine Learning, we now have the opportunity to bring scale and automation to the creation of realistic and enticing simulated content. In this work, we propose a framework to automate the generation of email and instant messaging-style group communications at scale. Such messaging platforms within organisations contain a lot of valuable information inside private communications and document attachments, making them an enticing target for an adversary. We address two key aspects of simulating this type of system: modelling when and with whom participants communicate, and generating topical, multi-party text to populate simulated conversation threads. We present the LogNormMix-Net Temporal Point Process as an approach to the first of these, building upon the intensity-free modeling approach of Shchur et al. to create a generative model for unicast and multi-cast communications. We demonstrate the use of fine-tuned, pre-trained language models to generate convincing multi-party conversation threads. A live email server is simulated by uniting our LogNormMix-Net TPP (to generate the communication timestamp, sender and recipients) with the language model, which generates the contents of the multi-party email threads. We evaluate the generated content with respect to a number of realism-based properties, that encourage a model to learn to generate content that will engage the attention of an adversary to achieve a deception outcome.



