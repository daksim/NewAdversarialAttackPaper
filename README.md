# Latest Adversarial Attack Papers
**update at 2022-09-20 06:31:25**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

## **1. Adversarial Driving: Attacking End-to-End Autonomous Driving**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2103.09151v4)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: As research in deep neural networks has advanced, deep convolutional networks have become feasible for automated driving tasks. In particular, there is an emerging trend of employing end-to-end neural network models for the automation of driving tasks. However, previous research has shown that deep neural network classifiers are vulnerable to adversarial attacks. For regression tasks, however, the effect of adversarial attacks is not as well understood. In this paper, we devise two white-box targeted attacks against end-to-end autonomous driving systems. The driving systems use a regression model that takes an image as input and outputs a steering angle. Our attacks manipulate the behavior of the autonomous driving system by perturbing the input image. Both attacks can be initiated in real-time on CPUs without employing GPUs. The efficiency of the attacks is illustrated using experiments conducted in Udacity. Demo video: https://youtu.be/I0i8uN2oOP0.



## **2. A Systematic Evaluation of Node Embedding Robustness**

cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.08064v1)

**Authors**: Alexandru Mara, Jefrey Lijffijt, Stephan Günnemann, Tijl De Bie

**Abstracts**: Node embedding methods map network nodes to low dimensional vectors that can be subsequently used in a variety of downstream prediction tasks. The popularity of these methods has significantly increased in recent years, yet, their robustness to perturbations of the input data is still poorly understood. In this paper, we assess the empirical robustness of node embedding models to random and adversarial poisoning attacks. Our systematic evaluation covers representative embedding methods based on Skip-Gram, matrix factorization, and deep neural networks. We compare edge addition, deletion and rewiring strategies computed using network properties as well as node labels. We also investigate the effect of label homophily and heterophily on robustness. We report qualitative results via embedding visualization and quantitative results in terms of downstream node classification and network reconstruction performances. We found that node classification suffers from higher performance degradation as opposed to network reconstruction, and that degree-based and label-based attacks are on average the most damaging.



## **3. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

cs.CR

Manuscript submitted to IEEE Trans. Dependable Secure Comput

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07936v1)

**Authors**: Zhuoruo Zhang, Chenyang Yu, He Huang, Rui Chang, Mingshuai Chen, Qinming Dai, Wenbo Shen, Yongwang Zhao, Kui Ren

**Abstracts**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.



## **4. SplitGuard: Detecting and Mitigating Training-Hijacking Attacks in Split Learning**

cs.CR

Proceedings of the 21st Workshop on Privacy in the Electronic Society  (WPES '22), November 7, 2022, Los Angeles, CA, USA

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2108.09052v3)

**Authors**: Ege Erdogan, Alptekin Kupcu, A. Ercument Cicek

**Abstracts**: Distributed deep learning frameworks such as split learning provide great benefits with regards to the computational cost of training deep neural networks and the privacy-aware utilization of the collective data of a group of data-holders. Split learning, in particular, achieves this goal by dividing a neural network between a client and a server so that the client computes the initial set of layers, and the server computes the rest. However, this method introduces a unique attack vector for a malicious server attempting to steal the client's private data: the server can direct the client model towards learning any task of its choice, e.g. towards outputting easily invertible values. With a concrete example already proposed (Pasquini et al., CCS '21), such training-hijacking attacks present a significant risk for the data privacy of split learning clients.   In this paper, we propose SplitGuard, a method by which a split learning client can detect whether it is being targeted by a training-hijacking attack or not. We experimentally evaluate our method's effectiveness, compare it with potential alternatives, and discuss in detail various points related to its use. We conclude that SplitGuard can effectively detect training-hijacking attacks while minimizing the amount of information recovered by the adversaries.



## **5. Privacy-Preserving Distributed Expectation Maximization for Gaussian Mixture Model using Subspace Perturbation**

cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07833v1)

**Authors**: Qiongxiu Li, Jaron Skovsted Gundersen, Katrine Tjell, Rafal Wisniewski, Mads Græsbøll Christensen

**Abstracts**: Privacy has become a major concern in machine learning. In fact, the federated learning is motivated by the privacy concern as it does not allow to transmit the private data but only intermediate updates. However, federated learning does not always guarantee privacy-preservation as the intermediate updates may also reveal sensitive information. In this paper, we give an explicit information-theoretical analysis of a federated expectation maximization algorithm for Gaussian mixture model and prove that the intermediate updates can cause severe privacy leakage. To address the privacy issue, we propose a fully decentralized privacy-preserving solution, which is able to securely compute the updates in each maximization step. Additionally, we consider two different types of security attacks: the honest-but-curious and eavesdropping adversary models. Numerical validation shows that the proposed approach has superior performance compared to the existing approach in terms of both the accuracy and privacy level.



## **6. A Large-scale Multiple-objective Method for Black-box Attack against Object Detection**

cs.CV

14 pages, 5 figures, ECCV2022

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07790v1)

**Authors**: Siyuan Liang, Longkang Li, Yanbo Fan, Xiaojun Jia, Jingzhi Li, Baoyuan Wu, Xiaochun Cao

**Abstracts**: Recent studies have shown that detectors based on deep models are vulnerable to adversarial examples, even in the black-box scenario where the attacker cannot access the model information. Most existing attack methods aim to minimize the true positive rate, which often shows poor attack performance, as another sub-optimal bounding box may be detected around the attacked bounding box to be the new true positive one. To settle this challenge, we propose to minimize the true positive rate and maximize the false positive rate, which can encourage more false positive objects to block the generation of new true positive bounding boxes. It is modeled as a multi-objective optimization (MOP) problem, of which the generic algorithm can search the Pareto-optimal. However, our task has more than two million decision variables, leading to low searching efficiency. Thus, we extend the standard Genetic Algorithm with Random Subset selection and Divide-and-Conquer, called GARSDC, which significantly improves the efficiency. Moreover, to alleviate the sensitivity to population quality in generic algorithms, we generate a gradient-prior initial population, utilizing the transferability between different detectors with similar backbones. Compared with the state-of-art attack methods, GARSDC decreases by an average 12.0 in the mAP and queries by about 1000 times in extensive experiments. Our codes can be found at https://github.com/LiangSiyuan21/ GARSDC.



## **7. PointCAT: Contrastive Adversarial Training for Robust Point Cloud Recognition**

cs.CV

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07788v1)

**Authors**: Qidong Huang, Xiaoyi Dong, Dongdong Chen, Hang Zhou, Weiming Zhang, Kui Zhang, Gang Hua, Nenghai Yu

**Abstracts**: Notwithstanding the prominent performance achieved in various applications, point cloud recognition models have often suffered from natural corruptions and adversarial perturbations. In this paper, we delve into boosting the general robustness of point cloud recognition models and propose Point-Cloud Contrastive Adversarial Training (PointCAT). The main intuition of PointCAT is encouraging the target recognition model to narrow the decision gap between clean point clouds and corrupted point clouds. Specifically, we leverage a supervised contrastive loss to facilitate the alignment and uniformity of the hypersphere features extracted by the recognition model, and design a pair of centralizing losses with the dynamic prototype guidance to avoid these features deviating from their belonging category clusters. To provide the more challenging corrupted point clouds, we adversarially train a noise generator along with the recognition model from the scratch, instead of using gradient-based attack as the inner loop like previous adversarial training methods. Comprehensive experiments show that the proposed PointCAT outperforms the baseline methods and dramatically boosts the robustness of different point cloud recognition models, under a variety of corruptions including isotropic point noises, the LiDAR simulated noises, random point dropping and adversarial perturbations.



## **8. On the Robustness of Graph Neural Diffusion to Topology Perturbations**

cs.LG

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.07754v1)

**Authors**: Yang Song, Qiyu Kang, Sijie Wang, Zhao Kai, Wee Peng Tay

**Abstracts**: Neural diffusion on graphs is a novel class of graph neural networks that has attracted increasing attention recently. The capability of graph neural partial differential equations (PDEs) in addressing common hurdles of graph neural networks (GNNs), such as the problems of over-smoothing and bottlenecks, has been investigated but not their robustness to adversarial attacks. In this work, we explore the robustness properties of graph neural PDEs. We empirically demonstrate that graph neural PDEs are intrinsically more robust against topology perturbation as compared to other GNNs. We provide insights into this phenomenon by exploiting the stability of the heat semigroup under graph topology perturbations. We discuss various graph diffusion operators and relate them to existing graph neural PDEs. Furthermore, we propose a general graph neural PDE framework based on which a new class of robust GNNs can be defined. We verify that the new model achieves comparable state-of-the-art performance on several benchmark datasets.



## **9. IPvSeeYou: Exploiting Leaked Identifiers in IPv6 for Street-Level Geolocation**

cs.NI

Accepted to S&P '23

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.06767v2)

**Authors**: Erik Rye, Robert Beverly

**Abstracts**: We present IPvSeeYou, a privacy attack that permits a remote and unprivileged adversary to physically geolocate many residential IPv6 hosts and networks with street-level precision. The crux of our method involves: 1) remotely discovering wide area (WAN) hardware MAC addresses from home routers; 2) correlating these MAC addresses with their WiFi BSSID counterparts of known location; and 3) extending coverage by associating devices connected to a common penultimate provider router.   We first obtain a large corpus of MACs embedded in IPv6 addresses via high-speed network probing. These MAC addresses are effectively leaked up the protocol stack and largely represent WAN interfaces of residential routers, many of which are all-in-one devices that also provide WiFi. We develop a technique to statistically infer the mapping between a router's WAN and WiFi MAC addresses across manufacturers and devices, and mount a large-scale data fusion attack that correlates WAN MACs with WiFi BSSIDs available in wardriving (geolocation) databases. Using these correlations, we geolocate the IPv6 prefixes of $>$12M routers in the wild across 146 countries and territories. Selected validation confirms a median geolocation error of 39 meters. We then exploit technology and deployment constraints to extend the attack to a larger set of IPv6 residential routers by clustering and associating devices with a common penultimate provider router. While we responsibly disclosed our results to several manufacturers and providers, the ossified ecosystem of deployed residential cable and DSL routers suggests that our attack will remain a privacy threat into the foreseeable future.



## **10. Adversarial Detection: Attacking Object Detection in Real Time**

cs.AI

7 pages, 10 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2209.01962v2)

**Authors**: Han Wu, Syed Yunas, Sareh Rowlands, Wenjie Ruan, Johan Wahlstrom

**Abstracts**: Intelligent robots rely on object detection models to perceive the environment. Following advances in deep learning security it has been revealed that object detection models are vulnerable to adversarial attacks. However, prior research primarily focuses on attacking static images or offline videos. Therefore, it is still unclear if such attacks could jeopardize real-world robotic applications in dynamic environments. This paper bridges this gap by presenting the first real-time online attack against object detection models. We devise three attacks that fabricate bounding boxes for nonexistent objects at desired locations. The attacks achieve a success rate of about 90% within about 20 iterations. The demo video is available at: https://youtu.be/zJZ1aNlXsMU.



## **11. A Man-in-the-Middle Attack against Object Detection Systems**

cs.RO

7 pages, 8 figures

**SubmitDate**: 2022-09-16    [paper-pdf](http://arxiv.org/pdf/2208.07174v2)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstracts**: Thanks to the increasing power of CPUs and GPUs in embedded systems, deep-learning-enabled object detection systems have become pervasive in a multitude of robotic applications. While deep learning models are vulnerable to several well-known adversarial attacks, the applicability of these attacks is severely limited by strict assumptions on, for example, access to the detection system. Inspired by Man-in-the-Middle attacks in cryptography, we propose a novel hardware attack on object detection systems that overcomes these limitations. Experiments prove that it is possible to generate an efficient Universal Adversarial Perturbation (UAP) within one minute and then use the perturbation to attack a detection system via the Man-in-the-Middle attack. These findings raise serious concerns for applications of deep learning models in safety-critical systems, such as autonomous driving. Demo Video: https://youtu.be/OvIpe-R3ZS8.



## **12. Adversarial Training for High-Stakes Reliability**

cs.LG

31 pages, 6 figures, fixed incorrect citation

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2205.01663v3)

**Authors**: Daniel M. Ziegler, Seraphina Nix, Lawrence Chan, Tim Bauman, Peter Schmidt-Nielsen, Tao Lin, Adam Scherlis, Noa Nabeshima, Ben Weinstein-Raun, Daniel de Haas, Buck Shlegeris, Nate Thomas

**Abstracts**: In the future, powerful AI systems may be deployed in high-stakes settings, where a single failure could be catastrophic. One technique for improving AI safety in high-stakes settings is adversarial training, which uses an adversary to generate examples to train on in order to achieve better worst-case performance.   In this work, we used a language generation task as a testbed for achieving high reliability through adversarial training. We created a series of adversarial training techniques -- including a tool that assists human adversaries -- to find and eliminate failures in a classifier that filters text completions suggested by a generator. In our simple "avoid injuries" task, we determined that we can set very conservative classifier thresholds without significantly impacting the quality of the filtered outputs. With our chosen thresholds, filtering with our baseline classifier decreases the rate of unsafe completions from about 2.4% to 0.003% on in-distribution data, which is near the limit of our ability to measure. We found that adversarial training significantly increased robustness to the adversarial attacks that we trained on, without affecting in-distribution performance. We hope to see further work in the high-stakes reliability setting, including more powerful tools for enhancing human adversaries and better ways to measure high levels of reliability, until we can confidently rule out the possibility of catastrophic deployment-time failures of powerful models.



## **13. How to Attack and Defend NextG Radio Access Network Slicing with Reinforcement Learning**

cs.NI

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2101.05768v2)

**Authors**: Yi Shi, Yalin E. Sagduyu, Tugba Erpek, M. Cenk Gursoy

**Abstracts**: In this paper, reinforcement learning (RL) for network slicing is considered in NextG radio access networks, where the base station (gNodeB) allocates resource blocks (RBs) to the requests of user equipments and aims to maximize the total reward of accepted requests over time. Based on adversarial machine learning, a novel over-the-air attack is introduced to manipulate the RL algorithm and disrupt NextG network slicing. The adversary observes the spectrum and builds its own RL based surrogate model that selects which RBs to jam subject to an energy budget with the objective of maximizing the number of failed requests due to jammed RBs. By jamming the RBs, the adversary reduces the RL algorithm's reward. As this reward is used as the input to update the RL algorithm, the performance does not recover even after the adversary stops jamming. This attack is evaluated in terms of both the recovery time and the (maximum and total) reward loss, and it is shown to be much more effective than benchmark (random and myopic) jamming attacks. Different reactive and proactive defense schemes (protecting the RL algorithm's updates or misleading the adversary's learning process) are introduced to show that it is viable to defend NextG network slicing against this attack.



## **14. A Light Recipe to Train Robust Vision Transformers**

cs.CV

Code available at https://github.com/dedeswim/vits-robustness-torch

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07399v1)

**Authors**: Edoardo Debenedetti, Vikash Sehwag, Prateek Mittal

**Abstracts**: In this paper, we ask whether Vision Transformers (ViTs) can serve as an underlying architecture for improving the adversarial robustness of machine learning models against evasion attacks. While earlier works have focused on improving Convolutional Neural Networks, we show that also ViTs are highly suitable for adversarial training to achieve competitive performance. We achieve this objective using a custom adversarial training recipe, discovered using rigorous ablation studies on a subset of the ImageNet dataset. The canonical training recipe for ViTs recommends strong data augmentation, in part to compensate for the lack of vision inductive bias of attention modules, when compared to convolutions. We show that this recipe achieves suboptimal performance when used for adversarial training. In contrast, we find that omitting all heavy data augmentation, and adding some additional bag-of-tricks ($\varepsilon$-warmup and larger weight decay), significantly boosts the performance of robust ViTs. We show that our recipe generalizes to different classes of ViT architectures and large-scale models on full ImageNet-1k. Additionally, investigating the reasons for the robustness of our models, we show that it is easier to generate strong attacks during training when using our recipe and that this leads to better robustness at test time. Finally, we further study one consequence of adversarial training by proposing a way to quantify the semantic nature of adversarial perturbations and highlight its correlation with the robustness of the model. Overall, we recommend that the community should avoid translating the canonical training recipes in ViTs to robust training and rethink common training choices in the context of adversarial training.



## **15. Continuous Patrolling Games**

cs.DM

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2008.07369v2)

**Authors**: Steve Alpern, Thuy Bui, Thomas Lidbetter, Katerina Papadaki

**Abstracts**: We study a patrolling game played on a network $Q$, considered as a metric space. The Attacker chooses a point of $Q$ (not necessarily a node) to attack during a chosen time interval of fixed duration. The Patroller chooses a unit speed path on $Q$ and intercepts the attack (and wins) if she visits the attacked point during the attack time interval. This zero-sum game models the problem of protecting roads or pipelines from an adversarial attack. The payoff to the maximizing Patroller is the probability that the attack is intercepted. Our results include the following: (i) a solution to the game for any network $Q$, as long as the time required to carry out the attack is sufficiently short, (ii) a solution to the game for all tree networks that satisfy a certain condition on their extremities, and (iii) a solution to the game for any attack duration for stars with one long arc and the remaining arcs equal in length. We present a conjecture on the solution of the game for arbitrary trees and establish it in certain cases.



## **16. Defending From Physically-Realizable Adversarial Attacks Through Internal Over-Activation Analysis**

cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2203.07341v2)

**Authors**: Giulio Rossolini, Federico Nesti, Fabio Brau, Alessandro Biondi, Giorgio Buttazzo

**Abstracts**: This work presents Z-Mask, a robust and effective strategy to improve the adversarial robustness of convolutional networks against physically-realizable adversarial attacks. The presented defense relies on specific Z-score analysis performed on the internal network features to detect and mask the pixels corresponding to adversarial objects in the input image. To this end, spatially contiguous activations are examined in shallow and deep layers to suggest potential adversarial regions. Such proposals are then aggregated through a multi-thresholding mechanism. The effectiveness of Z-Mask is evaluated with an extensive set of experiments carried out on models for both semantic segmentation and object detection. The evaluation is performed with both digital patches added to the input images and printed patches positioned in the real world. The obtained results confirm that Z-Mask outperforms the state-of-the-art methods in terms of both detection accuracy and overall performance of the networks under attack. Additional experiments showed that Z-Mask is also robust against possible defense-aware attacks.



## **17. Improving Robust Fairness via Balance Adversarial Training**

cs.LG

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2209.07534v1)

**Authors**: Chunyu Sun, Chenye Xu, Chengyuan Yao, Siyuan Liang, Yichao Wu, Ding Liang, XiangLong Liu, Aishan Liu

**Abstracts**: Adversarial training (AT) methods are effective against adversarial attacks, yet they introduce severe disparity of accuracy and robustness between different classes, known as the robust fairness problem. Previously proposed Fair Robust Learning (FRL) adaptively reweights different classes to improve fairness. However, the performance of the better-performed classes decreases, leading to a strong performance drop. In this paper, we observed two unfair phenomena during adversarial training: different difficulties in generating adversarial examples from each class (source-class fairness) and disparate target class tendencies when generating adversarial examples (target-class fairness). From the observations, we propose Balance Adversarial Training (BAT) to address the robust fairness problem. Regarding source-class fairness, we adjust the attack strength and difficulties of each class to generate samples near the decision boundary for easier and fairer model learning; considering target-class fairness, by introducing a uniform distribution constraint, we encourage the adversarial example generation process for each class with a fair tendency. Extensive experiments conducted on multiple datasets (CIFAR-10, CIFAR-100, and ImageNette) demonstrate that our method can significantly outperform other baselines in mitigating the robust fairness problem (+5-10\% on the worst class accuracy)



## **18. Decision-based Black-box Attack Against Vision Transformers via Patch-wise Adversarial Removal**

cs.CV

**SubmitDate**: 2022-09-15    [paper-pdf](http://arxiv.org/pdf/2112.03492v2)

**Authors**: Yucheng Shi, Yahong Han, Yu-an Tan, Xiaohui Kuang

**Abstracts**: Vision transformers (ViTs) have demonstrated impressive performance and stronger adversarial robustness compared to Convolutional Neural Networks (CNNs). On the one hand, ViTs' focus on global interaction between individual patches reduces the local noise sensitivity of images. On the other hand, the neglect of noise sensitivity differences between image regions by existing decision-based attacks further compromises the efficiency of noise compression, especially for ViTs. Therefore, validating the black-box adversarial robustness of ViTs when the target model can only be queried still remains a challenging problem. In this paper, we theoretically analyze the limitations of existing decision-based attacks from the perspective of noise sensitivity difference between regions of the image, and propose a new decision-based black-box attack against ViTs, termed Patch-wise Adversarial Removal (PAR). PAR divides images into patches through a coarse-to-fine search process and compresses the noise on each patch separately. PAR records the noise magnitude and noise sensitivity of each patch and selects the patch with the highest query value for noise compression. In addition, PAR can be used as a noise initialization method for other decision-based attacks to improve the noise compression efficiency on both ViTs and CNNs without introducing additional calculations. Extensive experiments on three datasets demonstrate that PAR achieves a much lower noise magnitude with the same number of queries.



## **19. PointACL:Adversarial Contrastive Learning for Robust Point Clouds Representation under Adversarial Attack**

cs.CV

arXiv admin note: text overlap with arXiv:2109.00179 by other authors

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06971v1)

**Authors**: Junxuan Huang, Yatong An, Lu cheng, Bai Chen, Junsong Yuan, Chunming Qiao

**Abstracts**: Despite recent success of self-supervised based contrastive learning model for 3D point clouds representation, the adversarial robustness of such pre-trained models raised concerns. Adversarial contrastive learning (ACL) is considered an effective way to improve the robustness of pre-trained models. In contrastive learning, the projector is considered an effective component for removing unnecessary feature information during contrastive pretraining and most ACL works also use contrastive loss with projected feature representations to generate adversarial examples in pretraining, while "unprojected " feature representations are used in generating adversarial inputs during inference.Because of the distribution gap between projected and "unprojected" features, their models are constrained of obtaining robust feature representations for downstream tasks. We introduce a new method to generate high-quality 3D adversarial examples for adversarial training by utilizing virtual adversarial loss with "unprojected" feature representations in contrastive learning framework. We present our robust aware loss function to train self-supervised contrastive learning framework adversarially. Furthermore, we find selecting high difference points with the Difference of Normal (DoN) operator as additional input for adversarial self-supervised contrastive learning can significantly improve the adversarial robustness of the pre-trained model. We validate our method, PointACL on downstream tasks, including 3D classification and 3D segmentation with multiple datasets. It obtains comparable robust accuracy over state-of-the-art contrastive adversarial learning methods.



## **20. Finetuning Pretrained Vision-Language Models with Correlation Information Bottleneck for Robust Visual Question Answering**

cs.CV

20 pages, 4 figures, 13 tables

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06954v1)

**Authors**: Jingjing Jiang, Ziyi Liu, Nanning Zheng

**Abstracts**: Benefiting from large-scale Pretrained Vision-Language Models (VL-PMs), the performance of Visual Question Answering (VQA) has started to approach human oracle performance. However, finetuning large-scale VL-PMs with limited data for VQA usually faces overfitting and poor generalization issues, leading to a lack of robustness. In this paper, we aim to improve the robustness of VQA systems (ie, the ability of the systems to defend against input variations and human-adversarial attacks) from the perspective of Information Bottleneck when finetuning VL-PMs for VQA. Generally, internal representations obtained by VL-PMs inevitably contain irrelevant and redundant information for the downstream VQA task, resulting in statistically spurious correlations and insensitivity to input variations. To encourage representations to converge to a minimal sufficient statistic in vision-language learning, we propose the Correlation Information Bottleneck (CIB) principle, which seeks a tradeoff between representation compression and redundancy by minimizing the mutual information (MI) between the inputs and internal representations while maximizing the MI between the outputs and the representations. Meanwhile, CIB measures the internal correlations among visual and linguistic inputs and representations by a symmetrized joint MI estimation. Extensive experiments on five VQA benchmarks of input robustness and two VQA benchmarks of human-adversarial robustness demonstrate the effectiveness and superiority of the proposed CIB in improving the robustness of VQA systems.



## **21. On the interplay of adversarial robustness and architecture components: patches, convolution and attention**

cs.CV

Presented at the "New Frontiers in Adversarial Machine Learning"  Workshop at ICML 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06953v1)

**Authors**: Francesco Croce, Matthias Hein

**Abstracts**: In recent years novel architecture components for image classification have been developed, starting with attention and patches used in transformers. While prior works have analyzed the influence of some aspects of architecture components on the robustness to adversarial attacks, in particular for vision transformers, the understanding of the main factors is still limited. We compare several (non)-robust classifiers with different architectures and study their properties, including the effect of adversarial training on the interpretability of the learnt features and robustness to unseen threat models. An ablation from ResNet to ConvNeXt reveals key architectural changes leading to almost $10\%$ higher $\ell_\infty$-robustness.



## **22. Robust Constrained Reinforcement Learning**

cs.LG

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06866v1)

**Authors**: Yue Wang, Fei Miao, Shaofeng Zou

**Abstracts**: Constrained reinforcement learning is to maximize the expected reward subject to constraints on utilities/costs. However, the training environment may not be the same as the test one, due to, e.g., modeling error, adversarial attack, non-stationarity, resulting in severe performance degradation and more importantly constraint violation. We propose a framework of robust constrained reinforcement learning under model uncertainty, where the MDP is not fixed but lies in some uncertainty set, the goal is to guarantee that constraints on utilities/costs are satisfied for all MDPs in the uncertainty set, and to maximize the worst-case reward performance over the uncertainty set. We design a robust primal-dual approach, and further theoretically develop guarantee on its convergence, complexity and robust feasibility. We then investigate a concrete example of $\delta$-contamination uncertainty set, design an online and model-free algorithm and theoretically characterize its sample complexity.



## **23. Certified Robustness to Word Substitution Ranking Attack for Neural Ranking Models**

cs.IR

Accepted by CIKM2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06691v1)

**Authors**: Chen Wu, Ruqing Zhang, Jiafeng Guo, Wei Chen, Yixing Fan, Maarten de Rijke, Xueqi Cheng

**Abstracts**: Neural ranking models (NRMs) have achieved promising results in information retrieval. NRMs have also been shown to be vulnerable to adversarial examples. A typical Word Substitution Ranking Attack (WSRA) against NRMs was proposed recently, in which an attacker promotes a target document in rankings by adding human-imperceptible perturbations to its text. This raises concerns when deploying NRMs in real-world applications. Therefore, it is important to develop techniques that defend against such attacks for NRMs. In empirical defenses adversarial examples are found during training and used to augment the training set. However, such methods offer no theoretical guarantee on the models' robustness and may eventually be broken by other sophisticated WSRAs. To escape this arms race, rigorous and provable certified defense methods for NRMs are needed.   To this end, we first define the \textit{Certified Top-$K$ Robustness} for ranking models since users mainly care about the top ranked results in real-world scenarios. A ranking model is said to be Certified Top-$K$ Robust on a ranked list when it is guaranteed to keep documents that are out of the top $K$ away from the top $K$ under any attack. Then, we introduce a Certified Defense method, named CertDR, to achieve certified top-$K$ robustness against WSRA, based on the idea of randomized smoothing. Specifically, we first construct a smoothed ranker by applying random word substitutions on the documents, and then leverage the ranking property jointly with the statistical property of the ensemble to provably certify top-$K$ robustness. Extensive experiments on two representative web search datasets demonstrate that CertDR can significantly outperform state-of-the-art empirical defense methods for ranking models.



## **24. Order-Disorder: Imitation Adversarial Attacks for Black-box Neural Ranking Models**

cs.IR

15 pages, 4 figures, accepted by ACM CCS 2022

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06506v1)

**Authors**: Jiawei Liu, Yangyang Kang, Di Tang, Kaisong Song, Changlong Sun, Xiaofeng Wang, Wei Lu, Xiaozhong Liu

**Abstracts**: Neural text ranking models have witnessed significant advancement and are increasingly being deployed in practice. Unfortunately, they also inherit adversarial vulnerabilities of general neural models, which have been detected but remain underexplored by prior studies. Moreover, the inherit adversarial vulnerabilities might be leveraged by blackhat SEO to defeat better-protected search engines. In this study, we propose an imitation adversarial attack on black-box neural passage ranking models. We first show that the target passage ranking model can be transparentized and imitated by enumerating critical queries/candidates and then train a ranking imitation model. Leveraging the ranking imitation model, we can elaborately manipulate the ranking results and transfer the manipulation attack to the target ranking model. For this purpose, we propose an innovative gradient-based attack method, empowered by the pairwise objective function, to generate adversarial triggers, which causes premeditated disorderliness with very few tokens. To equip the trigger camouflages, we add the next sentence prediction loss and the language model fluency constraint to the objective function. Experimental results on passage ranking demonstrate the effectiveness of the ranking imitation attack model and adversarial triggers against various SOTA neural ranking models. Furthermore, various mitigation analyses and human evaluation show the effectiveness of camouflages when facing potential mitigation approaches. To motivate other scholars to further investigate this novel and important problem, we make the experiment data and code publicly available.



## **25. Targeting interventions for displacement minimization in opinion dynamics**

cs.SI

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06481v1)

**Authors**: Luca Damonte, Giacomo Como, Fabio Fagnani

**Abstracts**: Social influence is largely recognized as a key factor in opinion formation processes. Recently, the role of external forces in inducing opinion displacement and polarization in social networks has attracted significant attention. This is in particular motivated by the necessity to understand and possibly prevent interference phenomena during political campaigns and elections. In this paper, we formulate and solve a targeted intervention problem for opinion displacement minimization on a social network. Specifically, we consider a min-max problem whereby a social planner (the defender) aims at selecting the optimal network intervention within her given budget constraint in order to minimize the opinion displacement in the system that an adversary (the attacker) is instead trying to maximize. Our results show that the optimal intervention of the defender has two regimes. For large enough budget, the optimal intervention of the social planner acts on all nodes proportionally to a new notion of network centrality. For lower budget values, such optimal intervention has a more delicate structure and is rather concentrated on a few target individuals.



## **26. Private Eye: On the Limits of Textual Screen Peeking via Eyeglass Reflections in Video Conferencing**

cs.CR

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2205.03971v2)

**Authors**: Yan Long, Chen Yan, Shilin Xiao, Shivan Prasad, Wenyuan Xu, Kevin Fu

**Abstracts**: Using mathematical modeling and human subjects experiments, this research explores the extent to which emerging webcams might leak recognizable textual and graphical information gleaming from eyeglass reflections captured by webcams. The primary goal of our work is to measure, compute, and predict the factors, limits, and thresholds of recognizability as webcam technology evolves in the future. Our work explores and characterizes the viable threat models based on optical attacks using multi-frame super resolution techniques on sequences of video frames. Our models and experimental results in a controlled lab setting show it is possible to reconstruct and recognize with over 75% accuracy on-screen texts that have heights as small as 10 mm with a 720p webcam. We further apply this threat model to web textual contents with varying attacker capabilities to find thresholds at which text becomes recognizable. Our user study with 20 participants suggests present-day 720p webcams are sufficient for adversaries to reconstruct textual content on big-font websites. Our models further show that the evolution towards 4K cameras will tip the threshold of text leakage to reconstruction of most header texts on popular websites. Besides textual targets, a case study on recognizing a closed-world dataset of Alexa top 100 websites with 720p webcams shows a maximum recognition accuracy of 94% with 10 participants even without using machine-learning models. Our research proposes near-term mitigations including a software prototype that users can use to blur the eyeglass areas of their video streams. For possible long-term defenses, we advocate an individual reflection testing procedure to assess threats under various settings, and justify the importance of following the principle of least privilege for privacy-sensitive scenarios.



## **27. TSFool: Crafting High-quality Adversarial Time Series through Multi-objective Optimization to Fool Recurrent Neural Network Classifiers**

cs.LG

9 pages, 5 figures

**SubmitDate**: 2022-09-14    [paper-pdf](http://arxiv.org/pdf/2209.06388v1)

**Authors**: Yanyun Wang, Dehui Du, Yuanhao Liu

**Abstracts**: Deep neural network (DNN) classifiers are vulnerable to adversarial attacks. Although the existing gradient-based attacks have achieved good performance in feed-forward model and image recognition tasks, the extension for time series classification in the recurrent neural network (RNN) remains a dilemma, because the cyclical structure of RNN prevents direct model differentiation and the visual sensitivity to perturbations of time series data challenges the traditional local optimization objective to minimize perturbation. In this paper, an efficient and widely applicable approach called TSFool for crafting high-quality adversarial time series for the RNN classifier is proposed. We propose a novel global optimization objective named Camouflage Coefficient to consider how well the adversarial samples hide in class clusters, and accordingly redefine the high-quality adversarial attack as a multi-objective optimization problem. We also propose a new idea to use intervalized weighted finite automata (IWFA) to capture deeply embedded vulnerable samples having otherness between features and latent manifold to guide the approximation to the optimization solution. Experiments on 22 UCR datasets are conducted to confirm that TSFool is a widely effective, efficient and high-quality approach with 93.22% less local perturbation, 32.33% better global camouflage, and 1.12 times speedup to existing methods.



## **28. PINCH: An Adversarial Extraction Attack Framework for Deep Learning Models**

cs.CR

15 pages, 11 figures, 2 tables

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.06300v1)

**Authors**: William Hackett, Stefan Trawicki, Zhengxin Yu, Neeraj Suri, Peter Garraghan

**Abstracts**: Deep Learning (DL) models increasingly power a diversity of applications. Unfortunately, this pervasiveness also makes them attractive targets for extraction attacks which can steal the architecture, parameters, and hyper-parameters of a targeted DL model. Existing extraction attack studies have observed varying levels of attack success for different DL models and datasets, yet the underlying cause(s) behind their susceptibility often remain unclear. Ascertaining such root-cause weaknesses would help facilitate secure DL systems, though this requires studying extraction attacks in a wide variety of scenarios to identify commonalities across attack success and DL characteristics. The overwhelmingly high technical effort and time required to understand, implement, and evaluate even a single attack makes it infeasible to explore the large number of unique extraction attack scenarios in existence, with current frameworks typically designed to only operate for specific attack types, datasets and hardware platforms. In this paper we present PINCH: an efficient and automated extraction attack framework capable of deploying and evaluating multiple DL models and attacks across heterogeneous hardware platforms. We demonstrate the effectiveness of PINCH by empirically evaluating a large number of previously unexplored extraction attack scenarios, as well as secondary attack staging. Our key findings show that 1) multiple characteristics affect extraction attack success spanning DL model architecture, dataset complexity, hardware, attack type, and 2) partially successful extraction attacks significantly enhance the success of further adversarial attack staging.



## **29. Certified Defences Against Adversarial Patch Attacks on Semantic Segmentation**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05980v1)

**Authors**: Maksym Yatsura, Kaspar Sakmann, N. Grace Hua, Matthias Hein, Jan Hendrik Metzen

**Abstracts**: Adversarial patch attacks are an emerging security threat for real world deep learning applications. We present Demasked Smoothing, the first approach (up to our knowledge) to certify the robustness of semantic segmentation models against this threat model. Previous work on certifiably defending against patch attacks has mostly focused on image classification task and often required changes in the model architecture and additional training which is undesirable and computationally expensive. In Demasked Smoothing, any segmentation model can be applied without particular training, fine-tuning, or restriction of the architecture. Using different masking strategies, Demasked Smoothing can be applied both for certified detection and certified recovery. In extensive experiments we show that Demasked Smoothing can on average certify 64% of the pixel predictions for a 1% patch in the detection task and 48% against a 0.5% patch for the recovery task on the ADE20K dataset.



## **30. Adversarial Inter-Group Link Injection Degrades the Fairness of Graph Neural Networks**

cs.LG

A shorter version of this work has been accepted by IEEE ICDM 2022

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05957v1)

**Authors**: Hussain Hussain, Meng Cao, Sandipan Sikdar, Denis Helic, Elisabeth Lex, Markus Strohmaier, Roman Kern

**Abstracts**: We present evidence for the existence and effectiveness of adversarial attacks on graph neural networks (GNNs) that aim to degrade fairness. These attacks can disadvantage a particular subgroup of nodes in GNN-based node classification, where nodes of the underlying network have sensitive attributes, such as race or gender. We conduct qualitative and experimental analyses explaining how adversarial link injection impairs the fairness of GNN predictions. For example, an attacker can compromise the fairness of GNN-based node classification by injecting adversarial links between nodes belonging to opposite subgroups and opposite class labels. Our experiments on empirical datasets demonstrate that adversarial fairness attacks can significantly degrade the fairness of GNN predictions (attacks are effective) with a low perturbation rate (attacks are efficient) and without a significant drop in accuracy (attacks are deceptive). This work demonstrates the vulnerability of GNN models to adversarial fairness attacks. We hope our findings raise awareness about this issue in our community and lay a foundation for the future development of GNN models that are more robust to such attacks.



## **31. An Evolutionary, Gradient-Free, Query-Efficient, Black-Box Algorithm for Generating Adversarial Instances in Deep Networks**

cs.CV

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.08297v2)

**Authors**: Raz Lapid, Zvika Haramaty, Moshe Sipper

**Abstracts**: Deep neural networks (DNNs) are sensitive to adversarial data in a variety of scenarios, including the black-box scenario, where the attacker is only allowed to query the trained model and receive an output. Existing black-box methods for creating adversarial instances are costly, often using gradient estimation or training a replacement network. This paper introduces \textbf{Qu}ery-Efficient \textbf{E}volutiona\textbf{ry} \textbf{Attack}, \textit{QuEry Attack}, an untargeted, score-based, black-box attack. QuEry Attack is based on a novel objective function that can be used in gradient-free optimization problems. The attack only requires access to the output logits of the classifier and is thus not affected by gradient masking. No additional information is needed, rendering our method more suitable to real-life situations. We test its performance with three different state-of-the-art models -- Inception-v3, ResNet-50, and VGG-16-BN -- against three benchmark datasets: MNIST, CIFAR10 and ImageNet. Furthermore, we evaluate QuEry Attack's performance on non-differential transformation defenses and state-of-the-art robust models. Our results demonstrate the superior performance of QuEry Attack, both in terms of accuracy score and query efficiency.



## **32. Bayesian Pseudo Labels: Expectation Maximization for Robust and Efficient Semi-Supervised Segmentation**

cs.CV

MICCAI 2022 (Early accept, Student Travel Award)

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2208.04435v3)

**Authors**: Mou-Cheng Xu, Yukun Zhou, Chen Jin, Marius de Groot, Daniel C. Alexander, Neil P. Oxtoby, Yipeng Hu, Joseph Jacob

**Abstracts**: This paper concerns pseudo labelling in segmentation. Our contribution is fourfold. Firstly, we present a new formulation of pseudo-labelling as an Expectation-Maximization (EM) algorithm for clear statistical interpretation. Secondly, we propose a semi-supervised medical image segmentation method purely based on the original pseudo labelling, namely SegPL. We demonstrate SegPL is a competitive approach against state-of-the-art consistency regularisation based methods on semi-supervised segmentation on a 2D multi-class MRI brain tumour segmentation task and a 3D binary CT lung vessel segmentation task. The simplicity of SegPL allows less computational cost comparing to prior methods. Thirdly, we demonstrate that the effectiveness of SegPL may originate from its robustness against out-of-distribution noises and adversarial attacks. Lastly, under the EM framework, we introduce a probabilistic generalisation of SegPL via variational inference, which learns a dynamic threshold for pseudo labelling during the training. We show that SegPL with variational inference can perform uncertainty estimation on par with the gold-standard method Deep Ensemble.



## **33. Adversarial Coreset Selection for Efficient Robust Training**

cs.LG

Extended version of the ECCV2022 paper: arXiv:2112.00378. arXiv admin  note: substantial text overlap with arXiv:2112.00378

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05785v1)

**Authors**: Hadi M. Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstracts**: Neural networks are vulnerable to adversarial attacks: adding well-crafted, imperceptible perturbations to their input can modify their output. Adversarial training is one of the most effective approaches to training robust models against such attacks. Unfortunately, this method is much slower than vanilla training of neural networks since it needs to construct adversarial examples for the entire training data at every iteration. By leveraging the theory of coreset selection, we show how selecting a small subset of training data provides a principled approach to reducing the time complexity of robust training. To this end, we first provide convergence guarantees for adversarial coreset selection. In particular, we show that the convergence bound is directly related to how well our coresets can approximate the gradient computed over the entire training data. Motivated by our theoretical analysis, we propose using this gradient approximation error as our adversarial coreset selection objective to reduce the training set size effectively. Once built, we run adversarial training over this subset of the training data. Unlike existing methods, our approach can be adapted to a wide variety of training objectives, including TRADES, $\ell_p$-PGD, and Perceptual Adversarial Training. We conduct extensive experiments to demonstrate that our approach speeds up adversarial training by 2-3 times while experiencing a slight degradation in the clean and robust accuracy.



## **34. Adaptive Perturbation Generation for Multiple Backdoors Detection**

cs.CV

7 pages, 5 figures

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05244v2)

**Authors**: Yuhang Wang, Huafeng Shi, Rui Min, Ruijia Wu, Siyuan Liang, Yichao Wu, Ding Liang, Aishan Liu

**Abstracts**: Extensive evidence has demonstrated that deep neural networks (DNNs) are vulnerable to backdoor attacks, which motivates the development of backdoor detection methods. Existing backdoor detection methods are typically tailored for backdoor attacks with individual specific types (e.g., patch-based or perturbation-based). However, adversaries are likely to generate multiple types of backdoor attacks in practice, which challenges the current detection strategies. Based on the fact that adversarial perturbations are highly correlated with trigger patterns, this paper proposes the Adaptive Perturbation Generation (APG) framework to detect multiple types of backdoor attacks by adaptively injecting adversarial perturbations. Since different trigger patterns turn out to show highly diverse behaviors under the same adversarial perturbations, we first design the global-to-local strategy to fit the multiple types of backdoor triggers via adjusting the region and budget of attacks. To further increase the efficiency of perturbation injection, we introduce a gradient-guided mask generation strategy to search for the optimal regions for adversarial attacks. Extensive experiments conducted on multiple datasets (CIFAR-10, GTSRB, Tiny-ImageNet) demonstrate that our method outperforms state-of-the-art baselines by large margins(+12%).



## **35. A Tale of HodgeRank and Spectral Method: Target Attack Against Rank Aggregation Is the Fixed Point of Adversarial Game**

cs.LG

33 pages,  https://github.com/alphaprime/Target_Attack_Rank_Aggregation

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05742v1)

**Authors**: Ke Ma, Qianqian Xu, Jinshan Zeng, Guorong Li, Xiaochun Cao, Qingming Huang

**Abstracts**: Rank aggregation with pairwise comparisons has shown promising results in elections, sports competitions, recommendations, and information retrieval. However, little attention has been paid to the security issue of such algorithms, in contrast to numerous research work on the computational and statistical characteristics. Driven by huge profits, the potential adversary has strong motivation and incentives to manipulate the ranking list. Meanwhile, the intrinsic vulnerability of the rank aggregation methods is not well studied in the literature. To fully understand the possible risks, we focus on the purposeful adversary who desires to designate the aggregated results by modifying the pairwise data in this paper. From the perspective of the dynamical system, the attack behavior with a target ranking list is a fixed point belonging to the composition of the adversary and the victim. To perform the targeted attack, we formulate the interaction between the adversary and the victim as a game-theoretic framework consisting of two continuous operators while Nash equilibrium is established. Then two procedures against HodgeRank and RankCentrality are constructed to produce the modification of the original data. Furthermore, we prove that the victims will produce the target ranking list once the adversary masters the complete information. It is noteworthy that the proposed methods allow the adversary only to hold incomplete information or imperfect feedback and perform the purposeful attack. The effectiveness of the suggested target attack strategies is demonstrated by a series of toy simulations and several real-world data experiments. These experimental results show that the proposed methods could achieve the attacker's goal in the sense that the leading candidate of the perturbed ranking list is the designated one by the adversary.



## **36. Sample Complexity of an Adversarial Attack on UCB-based Best-arm Identification Policy**

cs.LG

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.05692v1)

**Authors**: Varsha Pendyala

**Abstracts**: In this work I study the problem of adversarial perturbations to rewards, in a Multi-armed bandit (MAB) setting. Specifically, I focus on an adversarial attack to a UCB type best-arm identification policy applied to a stochastic MAB. The UCB attack presented in [1] results in pulling a target arm K very often. I used the attack model of [1] to derive the sample complexity required for selecting target arm K as the best arm. I have proved that the stopping condition of UCB based best-arm identification algorithm given in [2], can be achieved by the target arm K in T rounds, where T depends only on the total number of arms and $\sigma$ parameter of $\sigma^2-$ sub-Gaussian random rewards of the arms.



## **37. Replay-based Recovery for Autonomous Robotic Vehicles from Sensor Deception Attacks**

cs.RO

**SubmitDate**: 2022-09-13    [paper-pdf](http://arxiv.org/pdf/2209.04554v2)

**Authors**: Pritam Dash, Guanpeng Li, Mehdi Karimibiuki, Karthik Pattabiraman

**Abstracts**: Sensors are crucial for autonomous operation in robotic vehicles (RV). Physical attacks on sensors such as sensor tampering or spoofing can feed erroneous values to RVs through physical channels, which results in mission failures. In this paper, we present DeLorean, a comprehensive diagnosis and recovery framework for securing autonomous RVs from physical attacks. We consider a strong form of physical attack called sensor deception attacks (SDAs), in which the adversary targets multiple sensors of different types simultaneously (even including all sensors). Under SDAs, DeLorean inspects the attack induced errors, identifies the targeted sensors, and prevents the erroneous sensor inputs from being used in RV's feedback control loop. DeLorean replays historic state information in the feedback control loop and recovers the RV from attacks. Our evaluation on four real and two simulated RVs shows that DeLorean can recover RVs from different attacks, and ensure mission success in 94% of the cases (on average), without any crashes. DeLorean incurs low performance, memory and battery overheads.



## **38. Boosting Robustness Verification of Semantic Feature Neighborhoods**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05446v1)

**Authors**: Anan Kabaha, Dana Drachsler-Cohen

**Abstracts**: Deep neural networks have been shown to be vulnerable to adversarial attacks that perturb inputs based on semantic features. Existing robustness analyzers can reason about semantic feature neighborhoods to increase the networks' reliability. However, despite the significant progress in these techniques, they still struggle to scale to deep networks and large neighborhoods. In this work, we introduce VeeP, an active learning approach that splits the verification process into a series of smaller verification steps, each is submitted to an existing robustness analyzer. The key idea is to build on prior steps to predict the next optimal step. The optimal step is predicted by estimating the certification velocity and sensitivity via parametric regression. We evaluate VeeP on MNIST, Fashion-MNIST, CIFAR-10 and ImageNet and show that it can analyze neighborhoods of various features: brightness, contrast, hue, saturation, and lightness. We show that, on average, given a 90 minute timeout, VeeP verifies 96% of the maximally certifiable neighborhoods within 29 minutes, while existing splitting approaches verify, on average, 73% of the maximally certifiable neighborhoods within 58 minutes.



## **39. Bilevel Optimization with a Lower-level Contraction: Optimal Sample Complexity without Warm-Start**

stat.ML

35 pages, 2 figures. Code at  https://github.com/CSML-IIT-UCL/bioptexps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2202.03397v2)

**Authors**: Riccardo Grazzi, Massimiliano Pontil, Saverio Salzo

**Abstracts**: We analyze a general class of bilevel problems, in which the upper-level problem consists in the minimization of a smooth objective function and the lower-level problem is to find the fixed point of a smooth contraction map. This type of problems include instances of meta-learning, equilibrium models, hyperparameter optimization and data poisoning adversarial attacks. Several recent works have proposed algorithms which warm-start the lower level problem, i.e. they use the previous lower-level approximate solution as a staring point for the lower-level solver. This warm-start procedure allows one to improve the sample complexity in both the stochastic and deterministic settings, achieving in some cases the order-wise optimal sample complexity. However, there are situations, e.g., meta learning and equilibrium models, in which the warm-start procedure is not well-suited or ineffective. In this work we show that without warm-start, it is still possible to achieve order-wise optimal or near-optimal sample complexity. In particular, we propose a simple method which uses stochastic fixed point iterations at the lower-level and projected inexact gradient descent at the upper-level, that reaches an $\epsilon$-stationary point using $O(\epsilon^{-2})$ and $\tilde{O}(\epsilon^{-1})$ samples for the stochastic and the deterministic setting, respectively. Finally, compared to methods using warm-start, our approach yields a simpler analysis that does not need to study the coupled interactions between the upper-level and lower-level iterates



## **40. Fiat-Shamir for Proofs Lacks a Proof Even in the Presence of Shared Entanglement**

quant-ph

62 pages, 2 figures

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.02265v2)

**Authors**: Frédéric Dupuis, Philippe Lamontagne, Louis Salvail

**Abstracts**: We explore the cryptographic power of arbitrary shared physical resources. The most general such resource is access to a fresh entangled quantum state at the outset of each protocol execution. We call this the Common Reference Quantum State (CRQS) model, in analogy to the well-known Common Reference String (CRS). The CRQS model is a natural generalization of the CRS model but appears to be more powerful: in the two-party setting, a CRQS can sometimes exhibit properties associated with a Random Oracle queried once by measuring a maximally entangled state in one of many mutually unbiased bases. We formalize this notion as a Weak One-Time Random Oracle (WOTRO), where we only ask of the $m$--bit output to have some randomness when conditioned on the $n$--bit input.   We show that when $n-m\in\omega(\lg n)$, any protocol for WOTRO in the CRQS model can be attacked by an (inefficient) adversary. Moreover, our adversary is efficiently simulatable, which rules out the possibility of proving the computational security of a scheme by a black-box reduction to a cryptographic game assumption. On the other hand, we introduce a non-game quantum assumption for hash functions that implies WOTRO in the CRQ\$ model (where the CRQS consists only of EPR pairs). We first build a statistically secure WOTRO protocol where $m=n$, then hash the output.   The impossibility of WOTRO has the following consequences. First, we show the black-box impossibility of a quantum Fiat-Shamir transform, extending the impossibility result of Bitansky et al. (TCC '13) to the CRQS model. Second, we show a black-box impossibility result for a strenghtened version of quantum lightning (Zhandry, Eurocrypt '19) where quantum bolts have an additional parameter that cannot be changed without generating new bolts.



## **41. A Survey of Machine Unlearning**

cs.LG

fixed overlaps

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.02299v4)

**Authors**: Thanh Tam Nguyen, Thanh Trung Huynh, Phi Le Nguyen, Alan Wee-Chung Liew, Hongzhi Yin, Quoc Viet Hung Nguyen

**Abstracts**: Computer systems hold a large amount of personal data over decades. On the one hand, such data abundance allows breakthroughs in artificial intelligence (AI), especially machine learning (ML) models. On the other hand, it can threaten the privacy of users and weaken the trust between humans and AI. Recent regulations require that private information about a user can be removed from computer systems in general and from ML models in particular upon request (e.g. the "right to be forgotten"). While removing data from back-end databases should be straightforward, it is not sufficient in the AI context as ML models often "remember" the old data. Existing adversarial attacks proved that we can learn private membership or attributes of the training data from the trained models. This phenomenon calls for a new paradigm, namely machine unlearning, to make ML models forget about particular data. It turns out that recent works on machine unlearning have not been able to solve the problem completely due to the lack of common frameworks and resources. In this survey paper, we seek to provide a thorough investigation of machine unlearning in its definitions, scenarios, mechanisms, and applications. Specifically, as a categorical collection of state-of-the-art research, we hope to provide a broad reference for those seeking a primer on machine unlearning and its various formulations, design requirements, removal requests, algorithms, and uses in a variety of ML applications. Furthermore, we hope to outline key findings and trends in the paradigm as well as highlight new areas of research that have yet to see the application of machine unlearning, but could nonetheless benefit immensely. We hope this survey provides a valuable reference for ML researchers as well as those seeking to innovate privacy technologies. Our resources are at https://github.com/tamlhp/awesome-machine-unlearning.



## **42. GRNN: Generative Regression Neural Network -- A Data Leakage Attack for Federated Learning**

cs.LG

The source code can be found at: https://github.com/Rand2AI/GRNN

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2105.00529v3)

**Authors**: Hanchi Ren, Jingjing Deng, Xianghua Xie

**Abstracts**: Data privacy has become an increasingly important issue in Machine Learning (ML), where many approaches have been developed to tackle this challenge, e.g. cryptography (Homomorphic Encryption (HE), Differential Privacy (DP), etc.) and collaborative training (Secure Multi-Party Computation (MPC), Distributed Learning and Federated Learning (FL)). These techniques have a particular focus on data encryption or secure local computation. They transfer the intermediate information to the third party to compute the final result. Gradient exchanging is commonly considered to be a secure way of training a robust model collaboratively in Deep Learning (DL). However, recent researches have demonstrated that sensitive information can be recovered from the shared gradient. Generative Adversarial Network (GAN), in particular, has shown to be effective in recovering such information. However, GAN based techniques require additional information, such as class labels which are generally unavailable for privacy-preserved learning. In this paper, we show that, in the FL system, image-based privacy data can be easily recovered in full from the shared gradient only via our proposed Generative Regression Neural Network (GRNN). We formulate the attack to be a regression problem and optimize two branches of the generative model by minimizing the distance between gradients. We evaluate our method on several image classification tasks. The results illustrate that our proposed GRNN outperforms state-of-the-art methods with better stability, stronger robustness, and higher accuracy. It also has no convergence requirement to the global FL model. Moreover, we demonstrate information leakage using face re-identification. Some defense strategies are also discussed in this work.



## **43. Semantic-Preserving Adversarial Code Comprehension**

cs.CL

Accepted by COLING 2022

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05130v1)

**Authors**: Yiyang Li, Hongqiu Wu, Hai Zhao

**Abstracts**: Based on the tremendous success of pre-trained language models (PrLMs) for source code comprehension tasks, current literature studies either ways to further improve the performance (generalization) of PrLMs, or their robustness against adversarial attacks. However, they have to compromise on the trade-off between the two aspects and none of them consider improving both sides in an effective and practical way. To fill this gap, we propose Semantic-Preserving Adversarial Code Embeddings (SPACE) to find the worst-case semantic-preserving attacks while forcing the model to predict the correct labels under these worst cases. Experiments and analysis demonstrate that SPACE can stay robust against state-of-the-art attacks while boosting the performance of PrLMs for code.



## **44. Passive Triangulation Attack on ORide**

cs.CR

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2208.12216v2)

**Authors**: Shyam Murthy, Srinivas Vivek

**Abstracts**: Privacy preservation in Ride Hailing Services is intended to protect privacy of drivers and riders. ORide is one of the early RHS proposals published at USENIX Security Symposium 2017. In the ORide protocol, riders and drivers, operating in a zone, encrypt their locations using a Somewhat Homomorphic Encryption scheme (SHE) and forward them to the Service Provider (SP). SP homomorphically computes the squared Euclidean distance between riders and available drivers. Rider receives the encrypted distances and selects the optimal rider after decryption. In order to prevent a triangulation attack, SP randomly permutes the distances before sending them to the rider. In this work, we use propose a passive attack that uses triangulation to determine coordinates of all participating drivers whose permuted distances are available from the points of view of multiple honest-but-curious adversary riders. An attack on ORide was published at SAC 2021. The same paper proposes a countermeasure using noisy Euclidean distances to thwart their attack. We extend our attack to determine locations of drivers when given their permuted and noisy Euclidean distances from multiple points of reference, where the noise perturbation comes from a uniform distribution. We conduct experiments with different number of drivers and for different perturbation values. Our experiments show that we can determine locations of all drivers participating in the ORide protocol. For the perturbed distance version of the ORide protocol, our algorithm reveals locations of about 25% to 50% of participating drivers. Our algorithm runs in time polynomial in number of drivers.



## **45. CARE: Certifiably Robust Learning with Reasoning via Variational Inference**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.05055v1)

**Authors**: Jiawei Zhang, Linyi Li, Ce Zhang, Bo Li

**Abstracts**: Despite great recent advances achieved by deep neural networks (DNNs), they are often vulnerable to adversarial attacks. Intensive research efforts have been made to improve the robustness of DNNs; however, most empirical defenses can be adaptively attacked again, and the theoretically certified robustness is limited, especially on large-scale datasets. One potential root cause of such vulnerabilities for DNNs is that although they have demonstrated powerful expressiveness, they lack the reasoning ability to make robust and reliable predictions. In this paper, we aim to integrate domain knowledge to enable robust learning with the reasoning paradigm. In particular, we propose a certifiably robust learning with reasoning pipeline (CARE), which consists of a learning component and a reasoning component. Concretely, we use a set of standard DNNs to serve as the learning component to make semantic predictions, and we leverage the probabilistic graphical models, such as Markov logic networks (MLN), to serve as the reasoning component to enable knowledge/logic reasoning. However, it is known that the exact inference of MLN (reasoning) is #P-complete, which limits the scalability of the pipeline. To this end, we propose to approximate the MLN inference via variational inference based on an efficient expectation maximization algorithm. In particular, we leverage graph convolutional networks (GCNs) to encode the posterior distribution during variational inference and update the parameters of GCNs (E-step) and the weights of knowledge rules in MLN (M-step) iteratively. We conduct extensive experiments on different datasets and show that CARE achieves significantly higher certified robustness compared with the state-of-the-art baselines. We additionally conducted different ablation studies to demonstrate the empirical robustness of CARE and the effectiveness of different knowledge integration.



## **46. GFCL: A GRU-based Federated Continual Learning Framework against Data Poisoning Attacks in IoV**

cs.LG

11 pages, 12 figures, 3 tables; This work has been submitted to the  IEEE Transactions on Vehicular Technology for possible publication. Copyright  may be transferred without notice, after which this version may no longer be  accessible

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2204.11010v2)

**Authors**: Anum Talpur, Mohan Gurusamy

**Abstracts**: Integration of machine learning (ML) in 5G-based Internet of Vehicles (IoV) networks has enabled intelligent transportation and smart traffic management. Nonetheless, the security against adversarial poisoning attacks is also increasingly becoming a challenging task. Specifically, Deep Reinforcement Learning (DRL) is one of the widely used ML designs in IoV applications. The standard ML security techniques are not effective in DRL where the algorithm learns to solve sequential decision-making through continuous interaction with the environment, and the environment is time-varying, dynamic, and mobile. In this paper, we propose a Gated Recurrent Unit (GRU)-based federated continual learning (GFCL) anomaly detection framework against Sybil-based data poisoning attacks in IoV. The objective is to present a lightweight and scalable framework that learns and detects the illegitimate behavior without having a-priori training dataset consisting of attack samples. We use GRU to predict a future data sequence to analyze and detect illegitimate behavior from vehicles in a federated learning-based distributed manner. We investigate the performance of our framework using real-world vehicle mobility traces. The results demonstrate the effectiveness of our proposed solution in terms of different performance metrics.



## **47. Generate novel and robust samples from data: accessible sharing without privacy concerns**

cs.LG

**SubmitDate**: 2022-09-12    [paper-pdf](http://arxiv.org/pdf/2209.06113v1)

**Authors**: David Banh, Alan Huang

**Abstracts**: Generating new samples from data sets can mitigate extra expensive operations, increased invasive procedures, and mitigate privacy issues. These novel samples that are statistically robust can be used as a temporary and intermediate replacement when privacy is a concern. This method can enable better data sharing practices without problems relating to identification issues or biases that are flaws for an adversarial attack.



## **48. Resisting Deep Learning Models Against Adversarial Attack Transferability via Feature Randomization**

cs.CR

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2209.04930v1)

**Authors**: Ehsan Nowroozi, Mohammadreza Mohammadi, Pargol Golmohammadi, Yassine Mekdad, Mauro Conti, Selcuk Uluagac

**Abstracts**: In the past decades, the rise of artificial intelligence has given us the capabilities to solve the most challenging problems in our day-to-day lives, such as cancer prediction and autonomous navigation. However, these applications might not be reliable if not secured against adversarial attacks. In addition, recent works demonstrated that some adversarial examples are transferable across different models. Therefore, it is crucial to avoid such transferability via robust models that resist adversarial manipulations. In this paper, we propose a feature randomization-based approach that resists eight adversarial attacks targeting deep learning models in the testing phase. Our novel approach consists of changing the training strategy in the target network classifier and selecting random feature samples. We consider the attacker with a Limited-Knowledge and Semi-Knowledge conditions to undertake the most prevalent types of adversarial attacks. We evaluate the robustness of our approach using the well-known UNSW-NB15 datasets that include realistic and synthetic attacks. Afterward, we demonstrate that our strategy outperforms the existing state-of-the-art approach, such as the Most Powerful Attack, which consists of fine-tuning the network model against specific adversarial attacks. Finally, our experimental results show that our methodology can secure the target network and resists adversarial attack transferability by over 60%.



## **49. Detecting Adversarial Perturbations in Multi-Task Perception**

cs.CV

Accepted at IROS 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.01177v2)

**Authors**: Marvin Klingner, Varun Ravi Kumar, Senthil Yogamani, Andreas Bär, Tim Fingscheidt

**Abstracts**: While deep neural networks (DNNs) achieve impressive performance on environment perception tasks, their sensitivity to adversarial perturbations limits their use in practical applications. In this paper, we (i) propose a novel adversarial perturbation detection scheme based on multi-task perception of complex vision tasks (i.e., depth estimation and semantic segmentation). Specifically, adversarial perturbations are detected by inconsistencies between extracted edges of the input image, the depth output, and the segmentation output. To further improve this technique, we (ii) develop a novel edge consistency loss between all three modalities, thereby improving their initial consistency which in turn supports our detection scheme. We verify our detection scheme's effectiveness by employing various known attacks and image noises. In addition, we (iii) develop a multi-task adversarial attack, aiming at fooling both tasks as well as our detection scheme. Experimental evaluation on the Cityscapes and KITTI datasets shows that under an assumption of a 5% false positive rate up to 100% of images are correctly detected as adversarially perturbed, depending on the strength of the perturbation. Code is available at https://github.com/ifnspaml/AdvAttackDet. A short video at https://youtu.be/KKa6gOyWmH4 provides qualitative results.



## **50. Robustness through Cognitive Dissociation Mitigation in Contrastive Adversarial Training**

cs.LG

Accepted to ICMLC 2022

**SubmitDate**: 2022-09-11    [paper-pdf](http://arxiv.org/pdf/2203.08959v3)

**Authors**: Adir Rahamim, Itay Naeh

**Abstracts**: In this paper, we introduce a novel neural network training framework that increases model's adversarial robustness to adversarial attacks while maintaining high clean accuracy by combining contrastive learning (CL) with adversarial training (AT). We propose to improve model robustness to adversarial attacks by learning feature representations that are consistent under both data augmentations and adversarial perturbations. We leverage contrastive learning to improve adversarial robustness by considering an adversarial example as another positive example, and aim to maximize the similarity between random augmentations of data samples and their adversarial example, while constantly updating the classification head in order to avoid a cognitive dissociation between the classification head and the embedding space. This dissociation is caused by the fact that CL updates the network up to the embedding space, while freezing the classification head which is used to generate new positive adversarial examples. We validate our method, Contrastive Learning with Adversarial Features(CLAF), on the CIFAR-10 dataset on which it outperforms both robust accuracy and clean accuracy over alternative supervised and self-supervised adversarial learning methods.



